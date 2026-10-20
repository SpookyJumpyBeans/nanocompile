/* The two numbers a roofline needs, measured on this machine.
 *
 * nc_read(buf, chunks, floats_per_chunk): reads every float once, in
 * parallel chunks on nanocompile's pool, with enough independent vector
 * accumulators that the adds never limit it. Bytes / time is the read
 * bandwidth a weight-streaming decode step can hope for.
 *
 * nc_compute(out, tasks, iterations): register-only arithmetic, sixteen
 * independent vector chains of a multiply and a separate add, the
 * instruction mix nanocompile's kernels use (FMA contraction is off, so a
 * matmul's multiply-add is two instructions). Flops / time is the peak those
 * kernels can reach.
 */

#include <stdint.h>
#include <immintrin.h>

#ifdef _WIN32
#define NC_EXPORT __declspec(dllexport)
#else
#define NC_EXPORT __attribute__((visibility("default")))
#endif

typedef void (*nc_task)(void *, int64_t);
typedef void (*nc_runner)(nc_task, void *, int64_t);
static nc_runner run;
NC_EXPORT void nc_bind_pool(nc_runner r) { run = r; }

typedef struct { const float *buf; int64_t per; float *sums; } read_ctx;

static void read_chunk(void *c, int64_t i) {
    read_ctx *ctx = c;
    const float *p = ctx->buf + i * ctx->per;
    __m256 a = _mm256_setzero_ps(), b = a, d = a, e = a;
    for (int64_t j = 0; j < ctx->per; j += 32) {
        a = _mm256_add_ps(a, _mm256_loadu_ps(p + j));
        b = _mm256_add_ps(b, _mm256_loadu_ps(p + j + 8));
        d = _mm256_add_ps(d, _mm256_loadu_ps(p + j + 16));
        e = _mm256_add_ps(e, _mm256_loadu_ps(p + j + 24));
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a, b), _mm256_add_ps(d, e));
    float t[8];
    _mm256_storeu_ps(t, s);
    ctx->sums[i] = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];
}

NC_EXPORT void nc_read(const float *buf, int64_t chunks, int64_t per, float *sums) {
    read_ctx ctx = {buf, per, sums};
    run(read_chunk, &ctx, chunks);
}

typedef struct { int64_t iterations; float *out; } compute_ctx;

static void compute_task(void *c, int64_t i) {
    compute_ctx *ctx = c;
    __m256 m = _mm256_set1_ps(0.999f), k = _mm256_set1_ps(1e-3f);
    __m256 v[16];
    for (int r = 0; r < 16; ++r) v[r] = _mm256_set1_ps((float)(r + i));
    for (int64_t n = 0; n < ctx->iterations; ++n)
        for (int r = 0; r < 16; ++r) v[r] = _mm256_add_ps(_mm256_mul_ps(v[r], m), k);
    __m256 s = v[0];
    for (int r = 1; r < 16; ++r) s = _mm256_add_ps(s, v[r]);
    float t[8];
    _mm256_storeu_ps(t, s);
    ctx->out[i] = t[0];
}

/* Flops performed: tasks * iterations * 16 chains * 8 lanes * 2. */
NC_EXPORT void nc_compute(float *out, int64_t tasks, int64_t iterations) {
    compute_ctx ctx = {iterations, out};
    run(compute_task, &ctx, tasks);
}
