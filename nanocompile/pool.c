/* nanocompile's thread pool: persistent workers running one task at a time.
 *
 * A task is n independent iterations of fn(ctx, i). The caller runs
 * iterations too, and returns when all n are done. Workers are created once
 * and kept: nanoinfer measured a pool that spawned threads per call as 12x
 * slower than one thread, at decode sizes where a call is ~100 us of work.
 *
 * Iterations are claimed one at a time from a single 64-bit word holding the
 * task's generation and the next index, by compare-and-swap. Claiming
 * dynamically balances this CPU's fast and slow cores without knowing which
 * is which. The generation is what makes a late worker safe: a worker that
 * wakes after its task finished and a new one was posted fails its claim
 * (the generation moved on) instead of running the old function on the new
 * task's indices. The caller posts a new task only after every claimed
 * iteration of the last one has finished, so a successful claim pins the
 * task's function and context for as long as that iteration runs.
 *
 * Idle workers spin for a while before sleeping, because kernels arrive
 * back to back during a decode step and waking a sleeping thread costs tens
 * of microseconds on Windows.
 */

#include <stdint.h>
#include <stdatomic.h>

#ifdef _WIN32
#include <windows.h>
#define NC_EXPORT __declspec(dllexport)
static SRWLOCK lock = SRWLOCK_INIT;
static CONDITION_VARIABLE posted = CONDITION_VARIABLE_INIT;
static CONDITION_VARIABLE enlisted = CONDITION_VARIABLE_INIT;
#define LOCK() AcquireSRWLockExclusive(&lock)
#define UNLOCK() ReleaseSRWLockExclusive(&lock)
#define WAIT(c) SleepConditionVariableSRW(&(c), &lock, INFINITE, 0)
#define WAKE_ALL(c) WakeAllConditionVariable(&(c))
#else
#include <pthread.h>
#define NC_EXPORT __attribute__((visibility("default")))
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t posted = PTHREAD_COND_INITIALIZER;
static pthread_cond_t enlisted = PTHREAD_COND_INITIALIZER;
#define LOCK() pthread_mutex_lock(&lock)
#define UNLOCK() pthread_mutex_unlock(&lock)
#define WAIT(c) pthread_cond_wait(&(c), &lock)
#define WAKE_ALL(c) pthread_cond_broadcast(&(c))
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#include <immintrin.h>
#define PAUSE() _mm_pause()
#else
#define PAUSE() ((void)0)
#endif

#define MAX_WORKERS 63
#define SPIN 20000          /* pause instructions before an idle worker sleeps */

typedef void (*nc_task)(void *, int64_t);

static _Atomic(nc_task) task_fn;
static _Atomic(void *) task_ctx;
static _Atomic int64_t task_n;
static _Atomic uint64_t work;          /* generation << 32 | next index */
static _Atomic int64_t remaining;      /* iterations claimed or not, not yet done */
static _Atomic int sleepers;          /* active workers asleep, waiting for a task */
static _Atomic int active = 0;         /* workers taking part, besides the caller */
static int started = 0;                /* workers created */
static uint32_t generation = 0;

static int claim(uint32_t gen, int64_t *index) {
    uint64_t w = atomic_load_explicit(&work, memory_order_acquire);
    for (;;) {
        if ((uint32_t)(w >> 32) != gen)
            return 0;
        uint32_t i = (uint32_t)w;
        if ((int64_t)i >= atomic_load_explicit(&task_n, memory_order_relaxed))
            return 0;
        if (atomic_compare_exchange_weak_explicit(&work, &w, w + 1, memory_order_acq_rel,
                                                  memory_order_acquire)) {
            *index = i;
            return 1;
        }
    }
}

static void participate(uint32_t gen) {
    int64_t i;
    while (claim(gen, &i)) {
        nc_task fn = atomic_load_explicit(&task_fn, memory_order_relaxed);
        void *ctx = atomic_load_explicit(&task_ctx, memory_order_relaxed);
        fn(ctx, i);
        atomic_fetch_sub_explicit(&remaining, 1, memory_order_release);
    }
}

static uint32_t current_generation(void) {
    return (uint32_t)(atomic_load(&work) >> 32);
}

#ifdef _WIN32
static DWORD WINAPI worker(LPVOID arg)
#else
static void *worker(void *arg)
#endif
{
    int id = (int)(intptr_t)arg;
    uint32_t seen = current_generation();
    for (;;) {
        if (id >= atomic_load(&active)) {
            /* Not one of the threads in use: sleep until that changes. */
            LOCK();
            while (id >= atomic_load(&active))
                WAIT(enlisted);
            UNLOCK();
            seen = current_generation();
            continue;
        }
        uint32_t gen;
        int spins = 0;
        while ((gen = current_generation()) == seen && id < atomic_load(&active)) {
            if (++spins < SPIN) {
                PAUSE();
                continue;
            }
            LOCK();
            atomic_fetch_add(&sleepers, 1);
            while (current_generation() == seen && id < atomic_load(&active))
                WAIT(posted);
            atomic_fetch_sub(&sleepers, 1);
            UNLOCK();
            spins = 0;
        }
        if (gen == seen)
            continue;           /* dismissed while waiting */
        seen = gen;
        participate(gen);
    }
    return 0;
}

static void start_workers(int count) {
    while (started < count) {
        intptr_t id = started++;
#ifdef _WIN32
        HANDLE h = CreateThread(NULL, 0, worker, (LPVOID)id, 0, NULL);
        if (h) CloseHandle(h);
#else
        pthread_t t;
        if (pthread_create(&t, NULL, worker, (void *)id) == 0) pthread_detach(t);
#endif
    }
}

/* Threads taking part in a task, the caller included. 1 means no workers. */
NC_EXPORT void nc_pool_set_threads(int n) {
    if (n < 1) n = 1;
    if (n > MAX_WORKERS + 1) n = MAX_WORKERS + 1;
    start_workers(n - 1);
    LOCK();
    atomic_store(&active, n - 1);
    WAKE_ALL(enlisted);
    WAKE_ALL(posted);
    UNLOCK();
}

NC_EXPORT int nc_pool_threads(void) {
    return atomic_load(&active) + 1;
}

NC_EXPORT void nc_pool_run(nc_task fn, void *ctx, int64_t n) {
    if (n <= 0)
        return;
    if (atomic_load(&active) == 0 || n == 1 || n >= ((int64_t)1 << 32)) {
        for (int64_t i = 0; i < n; ++i) fn(ctx, i);
        return;
    }
    atomic_store_explicit(&task_fn, fn, memory_order_relaxed);
    atomic_store_explicit(&task_ctx, ctx, memory_order_relaxed);
    atomic_store_explicit(&task_n, n, memory_order_relaxed);
    atomic_store_explicit(&remaining, n, memory_order_relaxed);
    uint32_t gen = ++generation;
    atomic_store(&work, (uint64_t)gen << 32);   /* publishes the task */
    if (atomic_load(&sleepers) > 0) {
        LOCK();
        WAKE_ALL(posted);
        UNLOCK();
    }
    participate(gen);
    while (atomic_load_explicit(&remaining, memory_order_acquire) > 0)
        PAUSE();
}
