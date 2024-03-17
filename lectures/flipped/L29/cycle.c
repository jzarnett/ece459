/*

 setup    cycles   (est. overhead)  ~actual overhead

div [ALU] (100 Mi iterations)
 atomic: 20153782848 (20%)    ~ 3.8%
 mfence: 28202315112 (25%)    ~31.3%
vanilla: 19385020088

Reads:

TLB misses (64Mi iterations)
 atomic:  3776164048 (80%)   ~39.3%
 mfence: 12108883816 (50%)   ~81.1%
vanilla:  2293219400 

LLC misses (32Mi iterations)
 atomic:  3661686632 (40%)   ~23.3%
 mfence: 19596840824 (15%)   ~85.7%
vanilla:  2807258536

Writes:

TLB (64Mi iterations)
 atomic:  3864497496 (80%)  ~10.4%
 mfence: 13860666388 (50%)  ~75.0%
vanilla:  3461354848

LLC (32Mi iterations)
 atomic:  4023626584 (60%)  ~16.9%
 mfence: 21425039912 (20%)  ~84.4%
vanilla:  3345564432

*/

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#include "cycle.h"

/* xorshift128+ (http://xorshift.di.unimi.it/). */
#define RND(OUT, X, Y) do {                                             \
                uint64_t s1 = (X);                                      \
                const uint64_t s0 = (Y);                                \
                (X) = s0;                                               \
                s1 ^= s1 << 23;                                         \
                (OUT) = ((Y) = s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26)) + s0; \
        } while (0)

static uint64_t foo;

static uint32_t private[256];

__attribute__((noinline)) uint64_t
cache_misses(size_t n, uint64_t *buf, uint64_t mask)
{
        uint64_t state0 = 12345, state1 = 56789;
        uint64_t acc = 1244;
        uint64_t idx = 0;
        uint64_t x = 42;

        for (size_t i = 0; i < n; i++) {
                idx &= mask;
#ifdef WRITE
                acc = buf[idx] = acc ^ buf[idx];
#else
                acc ^= buf[idx];
#endif
                RND(idx, state0, state1);
                foo++;
                asm volatile(
#if defined(FENCE)
                        "mfence\n\t"
#elif defined(ATOMIC)
                        "lock incb %1\n\t"
#else
                        "addq $1, %%rax\n\t"
#endif
                        "addq $1, %%rax\n\t"
                        "addq $1, %%rax\n\t"
                        "addq $1, %%rax"
                        : "+a"(x) : "m"(private[128])
                        :"memory", "cc");

                idx &= mask;
#ifdef WRITE
                acc = buf[idx] = acc ^ buf[idx];
#else
                acc ^= buf[idx];
#endif
                RND(idx, state0, state1);

                asm volatile(
#if defined(FENCE)
                        "mfence\n\t"
#elif defined(ATOMIC)
                        "lock decb %1\n\t"
#else
                        "addq $1, %%rax\n\t"
#endif
                        "addq $1, %%rax\n\t"
                        "addq $1, %%rax\n\t"
                        "addq $1, %%rax"
                        : "+a"(x) : "m"(private[128])
                        :"memory", "cc");
                foo--;
        }

        return acc ^ x;
}

__attribute__((noinline)) uint64_t
div(size_t n)
{
        uint64_t acc = 0;

        for (size_t i = 0; i < n; i++) {
                asm volatile (
                        "movl $42, %%edx\n\t"
                        "movl $2343451, %%ebx\n\t"
                        "divq %%rbx\n\t"
                        "xorq %%rdx, %%rax\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
#if defined(FENCE)
                        "mfence\n\t"
#elif defined(ATOMIC)
                        "lock incb %1\n\t"
#else
                        "addl $1, %%ebx\n\t"
#endif
                        : "+a"(acc) : "m"(private[128])
                        : "memory", "cc", "rbx", "rdx");

                asm volatile (
                        "movl $42, %%edx\n\t"
                        "movl $2343451, %%ebx\n\t"
                        "divq %%rbx\n\t"
                        "xorq %%rdx, %%rax\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
                        "addl $1, %%ebx\n\t"
#if defined(FENCE)
                        "mfence\n\t"
#elif defined(ATOMIC)
                        "lock incb %1\n\t"
#else
                        "addl $1, %%ebx\n\t"
#endif
                        : "+a"(acc) : "m"(private[128])
                        : "memory", "cc", "rbx", "rdx");
        }

        return acc;
}

#ifndef MAP_HUGE_1GB
# define MAP_HUGE_1GB (30 << 26)
#endif

int
main ()
{
        uint64_t *buf;
        ticks begin;

#if defined(DIV)
        /* nothing to init.*/
        (void)buf;
#elif defined(TLB)
        buf = mmap(NULL, sizeof(uint64_t) << 26,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS /* | MAP_HUGETLB */,
                   -1, 0);

        madvise(buf, sizeof(uint64_t) << 26, MADV_UNMERGEABLE);
        madvise(buf, sizeof(uint64_t) << 26, MADV_NOHUGEPAGE);

        memset(buf, 42, sizeof(uint64_t) << 26);
#else
        buf = mmap(NULL, 1UL << 30,
                   PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB,
                   -1, 0);
        memset(buf, 42, 1UL << 30);
#endif


        begin = getticks();

#if defined(DIV)
        div(1024*1024*100);
#elif defined(TLB)
        cache_misses(64*1024*1024, buf, ((1UL << 20) - 1) & (-4096UL));
#else
        cache_misses(32*1024*1024, buf, ((1UL << 27) - 1));
#endif

        printf("elapsed %.f\n", elapsed(getticks(), begin));

        return 0;
}
