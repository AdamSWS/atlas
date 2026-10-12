// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
// Device side of the two-rank one-shot RDMA protocol, shared by
// rdma_oneshot_bf16 (rdma_oneshot.cu, which documents the protocol) and the
// kernels that land a send-only op themselves (rdma_oneshot_bf16 with
// add == ONESHOT_SEND_ONLY stages and publishes, and its consumer waits for
// the peer with oneshot_land_wait, reads the receive slot and advances the
// sequence with oneshot_land_done; ATLAS_GLM_HC_SEAM_AR_FOLD).
#include <cuda_bf16.h>
#include <stdint.h>

#define ONESHOT_POISON_DESYNC (1ull << 61)
#define ONESHOT_POISON_TIMEOUT (1ull << 62)
#define ONESHOT_POISON_MISMATCH (1ull << 63)
#define ONESHOT_FLAG_STRIDE 8 // u64 words: one 64-byte line per rail
#define ONESHOT_UNROLL 4       // receive loads in flight per thread

__device__ __forceinline__ uint64_t oneshot_ld_acquire(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ uint64_t oneshot_ld_acquire_gpu(const uint64_t* p) {
    uint64_t v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// Receive-slot loads: system-scope (never the non-coherent path), and without
// a memory clobber so a thread keeps several in flight.
__device__ __forceinline__ uint4 oneshot_ld_v4(const uint4* p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned short oneshot_ld_u16(const unsigned short* p) {
    unsigned short v;
    asm volatile("ld.relaxed.sys.global.u16 %0, [%1];" : "=h"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ uint32_t oneshot_ld_acquire_u32(const uint32_t* p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

// Staging stores into pinned host memory read by the NIC: write-through
// (mode 1) or plain (mode 2).
__device__ __forceinline__ void oneshot_st_v4(uint4* p, uint4 v, uint32_t mode) {
    if (mode == 1) {
        asm volatile("st.global.wt.v4.u32 [%0], {%1, %2, %3, %4};"
                     ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
    } else {
        *p = v;
    }
}

__device__ __forceinline__ void oneshot_st_u16(unsigned short* p, unsigned short v, uint32_t mode) {
    if (mode == 1) {
        asm volatile("st.global.wt.u16 [%0], %1;" ::"l"(p), "h"(v) : "memory");
    } else {
        *p = v;
    }
}

__device__ __forceinline__ uint64_t oneshot_now() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void oneshot_poison(uint64_t* poison, uint64_t word) {
    *(volatile uint64_t*)poison = word;
    __threadfence_system();
    __trap();
}

// Whether every rail's flag has reached `seq` (poisons on a size mismatch or
// a flag from beyond the next op).
__device__ __forceinline__ bool oneshot_arrived(
    const uint64_t* flags, uint32_t rails, uint64_t seq, uint32_t bytes, uint64_t* poison) {
    for (uint32_t r = 0; r < rails; ++r) {
        const uint64_t f = oneshot_ld_acquire(flags + r * ONESHOT_FLAG_STRIDE);
        if ((f >> 24) < seq) {
            return false;
        }
        if ((f >> 24) == seq && (f & 0xffffff) != bytes) {
            oneshot_poison(poison, ONESHOT_POISON_MISMATCH | seq);
        }
        if ((f >> 24) > seq + 1) {
            oneshot_poison(poison, ONESHOT_POISON_DESYNC | seq);
        }
    }
    return true;
}

// `add` of rdma_oneshot_bf16: stage and publish only; a consumer kernel lands.
#define ONESHOT_SEND_ONLY 2u

// Thread 0 of a CTA: wait until every rail's flag from the peer reaches op
// `seq`. The `poller` CTA reads the host flags each spin (and the poison word)
// and, when `publish`, republishes `seq` on the device go word (state[2]) for
// the others, which read go and the host flags every 256 spins, so no CTA
// depends on another's residency. Poisons and traps on a size mismatch, a
// flag beyond the next op, or a peer that never arrives within `timeout_ns`.
__device__ __forceinline__ void oneshot_land_wait(
    const uint64_t* flags, uint32_t rails, uint64_t* state, uint64_t seq, uint32_t bytes,
    uint64_t* poison, uint64_t timeout_ns, bool poller, bool publish) {
    uint64_t* go = state + 2;
    const uint64_t t0 = oneshot_now();
    for (uint32_t spin = 0;; ++spin) {
        const bool own = poller || spin % 256 == 255;
        if (own ? oneshot_arrived(flags, rails, seq, bytes, poison)
                : oneshot_ld_acquire_gpu(go) >= seq) {
            break;
        }
        if (spin % 64 == 63) {
            if (poller && oneshot_ld_acquire(poison) != 0) {
                __trap();
            }
            if (timeout_ns != 0 && oneshot_now() - t0 > timeout_ns) {
                oneshot_poison(poison, ONESHOT_POISON_TIMEOUT | seq);
            }
        }
    }
    if (poller && publish) {
        asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(go), "l"(seq) : "memory");
    }
}

// Thread 0 of a CTA, once its CTA has read the receive slot (after a CTA
// barrier): the last of the op's `ctas` CTAs advances the sequence.
__device__ __forceinline__ void oneshot_land_done(uint64_t* state, uint64_t seq, uint32_t ctas) {
    uint32_t* done = reinterpret_cast<uint32_t*>(state + 1);
    __threadfence();
    if (atomicAdd(done, 1u) == ctas - 1) {
        *done = 0;
        *(volatile uint64_t*)state = seq;
    }
}
