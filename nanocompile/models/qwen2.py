"""Qwen2 written in the frontend, and a driver that runs it a step at a time.

The graph is the whole forward pass, embedding to logits, for one step. Two
symbols size it: ``seq``, the tokens in this step, and ``past``, the tokens
already in the KV cache. The prompt is a step with ``past = 0`` and decode is
``seq = 1``, and both are the same graph. That mirrors nanoinfer's choice to
have one forward function whether or not a cache is present, for the same
reason: the claim that cached and uncached compute the same thing is only
checkable if there is one thing.

The cache lives outside the graph. Past keys and values are inputs, the keys
and values of the new tokens are outputs, and :class:`KVCache` stitches them
together between steps. Phase 7 moves that buffer into the compiled library;
until then the driver owns it.

Weights are bound by their names in the HuggingFace checkpoint
(``model.layers.0.self_attn.q_proj.weight``), so a weight can be traced from
the file to the IR dump without a translation table.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nanocompile import dtypes, interpreter
from nanocompile.frontend import GraphBuilder, Tensor, concat, gather
from nanocompile.ir import Graph
from nanocompile.nn import (
    apply_rope,
    causal_mask,
    linear,
    repeat_kv,
    rms_norm,
    rope_tables,
    silu,
    softmax,
)


@dataclass(frozen=True)
class Qwen2Config:
    """The hyperparameters the graph's shape depends on."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must divide evenly into attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must divide evenly into KV heads")
        if self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_group_size(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, data: Mapping) -> "Qwen2Config":
        if data.get("architectures") != ["Qwen2ForCausalLM"]:
            raise ValueError(f"not a Qwen2 config: {data.get('architectures')}")
        if data.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {data['hidden_act']!r} is not implemented")
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            num_key_value_heads=int(data.get("num_key_value_heads", data["num_attention_heads"])),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data.get("rope_theta", 10000.0)),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
        )

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "Qwen2Config":
        path = Path(model_dir) / "config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# -- the graph ---------------------------------------------------------------


def _split_heads(x: Tensor, heads: int, head_dim: int) -> Tensor:
    """``[seq, heads * d] -> [heads, seq, d]``: split the feature axis, then transpose."""
    seq = x.shape[0]
    return x.reshape(seq, heads, head_dim).permute(1, 0, 2)


def _merge_heads(x: Tensor) -> Tensor:
    heads, seq, head_dim = x.shape
    return x.permute(1, 0, 2).reshape(seq, heads * head_dim)


def _attention(
    b: GraphBuilder,
    config: Qwen2Config,
    layer: int,
    x: Tensor,
    rope: tuple[Tensor, Tensor],
    mask: Tensor,
    past_keys: Tensor,
    past_values: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Grouped-query attention. Returns the output and this step's keys and values."""
    p = f"model.layers.{layer}.self_attn"
    hidden, d = config.hidden_size, config.head_dim
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads

    def projection(name: str, out_features: int) -> Tensor:
        weight = b.weight(f"{p}.{name}.weight", (out_features, hidden))
        bias = b.weight(f"{p}.{name}.bias", (out_features,))
        return linear(x, weight, bias)

    # Qwen2 has biases on Q, K and V and none on the output projection.
    q = _split_heads(projection("q_proj", heads * d), heads, d)
    k = _split_heads(projection("k_proj", kv_heads * d), kv_heads, d)
    v = _split_heads(projection("v_proj", kv_heads * d), kv_heads, d)

    # Keys are cached after rotation: a token's position never changes.
    cos, sin = rope
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    past = past_keys.shape[2]
    k_all = concat([past_keys.slice(0, layer, layer + 1).reshape(kv_heads, past, d), k], axis=1)
    v_all = concat([past_values.slice(0, layer, layer + 1).reshape(kv_heads, past, d), v], axis=1)

    k_rep = repeat_kv(k_all, config.kv_group_size)
    v_rep = repeat_kv(v_all, config.kv_group_size)

    scores = (q @ k_rep.swapaxes(-1, -2)) * np.float32(d**-0.5)
    weights = softmax(scores + mask, axis=-1)
    context = _merge_heads(weights @ v_rep)

    out = linear(context, b.weight(f"{p}.o_proj.weight", (hidden, heads * d)))
    return out, k, v


def _feed_forward(b: GraphBuilder, config: Qwen2Config, layer: int, x: Tensor) -> Tensor:
    """SwiGLU: ``down(silu(gate(x)) * up(x))``."""
    p = f"model.layers.{layer}.mlp"
    hidden, inter = config.hidden_size, config.intermediate_size
    gate = linear(x, b.weight(f"{p}.gate_proj.weight", (inter, hidden)))
    up = linear(x, b.weight(f"{p}.up_proj.weight", (inter, hidden)))
    return linear(silu(gate) * up, b.weight(f"{p}.down_proj.weight", (hidden, inter)))


def build_qwen2(config: Qwen2Config, last_only: bool = False) -> Graph:
    """The forward pass for one step.

    Inputs: ``token_ids: i64[seq]``, and ``past_keys`` / ``past_values``,
    each ``f32[layers, kv_heads, past, head_dim]``.

    Outputs: ``logits`` (``[seq, vocab]``, or ``[1, vocab]`` with
    ``last_only``), and ``keys`` / ``values`` for the ``seq`` new tokens, each
    ``[layers, kv_heads, seq, head_dim]``.

    ``last_only`` slices before the output projection rather than after. That
    projection is ``[seq, 896] @ [896, 151936]``, the largest matmul in the
    model, and generation only reads its last row.
    """
    b = GraphBuilder("qwen2_last" if last_only else "qwen2")
    seq, past = b.symbol("seq"), b.symbol("past")
    layers, kv_heads, d = config.num_hidden_layers, config.num_key_value_heads, config.head_dim
    hidden_size, eps = config.hidden_size, config.rms_norm_eps

    token_ids = b.input("token_ids", dtypes.i64, (seq,))
    past_keys = b.input("past_keys", dtypes.f32, (layers, kv_heads, past, d))
    past_values = b.input("past_values", dtypes.f32, (layers, kv_heads, past, d))

    # Positions continue from the cache: new token i is at past + i.
    positions = b.iota((seq,), axis=0) + b.dim(past)
    rope = rope_tables(positions, d, config.rope_theta)
    mask = causal_mask(b, seq, past)

    embed = b.weight("model.embed_tokens.weight", (config.vocab_size, hidden_size))
    x = gather(embed, token_ids)

    new_keys, new_values = [], []
    for layer in range(layers):
        p = f"model.layers.{layer}"
        normed = rms_norm(x, b.weight(f"{p}.input_layernorm.weight", (hidden_size,)), eps)
        attended, k, v = _attention(b, config, layer, normed, rope, mask, past_keys, past_values)
        x = x + attended
        normed = rms_norm(x, b.weight(f"{p}.post_attention_layernorm.weight", (hidden_size,)), eps)
        x = x + _feed_forward(b, config, layer, normed)
        new_keys.append(k.reshape(1, kv_heads, seq, d))
        new_values.append(v.reshape(1, kv_heads, seq, d))

    x = rms_norm(x, b.weight("model.norm.weight", (hidden_size,)), eps)
    if last_only:
        x = x.slice(0, seq - 1, seq)

    if config.tie_word_embeddings:
        lm_head = embed  # the same node: one weight, read twice
    else:
        lm_head = b.weight("lm_head.weight", (config.vocab_size, hidden_size))

    return b.build(
        {
            "logits": linear(x, lm_head),
            "keys": concat(new_keys, axis=0),
            "values": concat(new_values, axis=0),
        }
    )


# -- running it --------------------------------------------------------------

Runner = Callable[[Graph, Mapping[str, np.ndarray], Mapping[str, np.ndarray]], dict[str, np.ndarray]]


class KVCache:
    """Keys and values for one sequence, preallocated and grown by doubling.

    Appending by concatenation would copy the whole cache every step, which is
    the cost a cache exists to avoid. The graph receives a view of the filled
    region, ``[layers, kv_heads, length, head_dim]``.
    """

    def __init__(self, config: Qwen2Config, capacity: int = 256) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.config = config
        self.length = 0
        self._keys = self._allocate(capacity)
        self._values = self._allocate(capacity)

    def _allocate(self, capacity: int) -> np.ndarray:
        c = self.config
        return np.zeros(
            (c.num_hidden_layers, c.num_key_value_heads, capacity, c.head_dim), dtype=np.float32
        )

    @property
    def capacity(self) -> int:
        return self._keys.shape[2]

    def past(self) -> tuple[np.ndarray, np.ndarray]:
        return self._keys[:, :, : self.length], self._values[:, :, : self.length]

    def append(self, keys: np.ndarray, values: np.ndarray) -> None:
        n = keys.shape[2]
        if self.length + n > self.capacity:
            capacity = max(self.length + n, 2 * self.capacity)
            for name in ("_keys", "_values"):
                grown = self._allocate(capacity)
                grown[:, :, : self.length] = getattr(self, name)[:, :, : self.length]
                setattr(self, name, grown)
        self._keys[:, :, self.length : self.length + n] = keys
        self._values[:, :, self.length : self.length + n] = values
        self.length += n


class Qwen2:
    """The compiled model's front door: token IDs in, logits out.

    ``run`` executes a graph. In phase 1 it is the interpreter; from phase 2
    on it is generated code, and nothing else here changes.
    """

    def __init__(
        self,
        config: Qwen2Config,
        weights: Mapping[str, np.ndarray],
        run: Runner = interpreter.run,
    ) -> None:
        self.config = config
        self.weights = weights
        self.run = run
        self.graph = build_qwen2(config)
        self.graph_last = build_qwen2(config, last_only=True)

    def new_cache(self, capacity: int = 256) -> KVCache:
        return KVCache(self.config, capacity)

    def forward(
        self,
        token_ids: Sequence[int] | np.ndarray,
        cache: KVCache | None = None,
        last_only: bool = False,
    ) -> np.ndarray:
        """Logits for ``token_ids``, the tokens not already in ``cache``.

        Without a cache this is the whole sequence from position 0.
        """
        token_ids = np.asarray(token_ids, dtype=np.int64)
        if token_ids.ndim != 1 or token_ids.size == 0:
            raise ValueError(f"expected a non-empty 1-D sequence of token IDs, got {token_ids.shape}")

        if cache is None:
            c = self.config
            empty = np.zeros((c.num_hidden_layers, c.num_key_value_heads, 0, c.head_dim), np.float32)
            past_keys = past_values = empty
        else:
            past_keys, past_values = cache.past()

        graph = self.graph_last if last_only else self.graph
        out = self.run(
            graph,
            {"token_ids": token_ids, "past_keys": past_keys, "past_values": past_values},
            self.weights,
        )
        if cache is not None:
            cache.append(out["keys"], out["values"])
        return out["logits"]


def greedy(
    model: Qwen2,
    prompt_ids: Sequence[int],
    max_new_tokens: int,
    use_cache: bool = True,
) -> list[int]:
    """Argmax decoding. The only kind two engines can be compared on token for token."""
    if not prompt_ids:
        raise ValueError("cannot generate from an empty prompt")
    ids = list(prompt_ids)
    cache = model.new_cache(len(ids) + max_new_tokens) if use_cache else None
    pending = ids

    generated: list[int] = []
    for _ in range(max_new_tokens):
        logits = model.forward(pending, cache=cache, last_only=True)
        token = int(np.argmax(logits[0]))
        generated.append(token)
        ids.append(token)
        pending = [token] if use_cache else ids
    return generated
