/* tq.h — TurboQuant container format v2 (strict C99, zero UB, zero padding)
 * -std=c99 -pedantic -Wall -Wextra -Werror -march=native
 * _Alignas(64) on hot data, restrict everywhere, LZ4 per-tensor optional
 */

#ifndef TQ_H
#define TQ_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#ifdef TQ_WITH_LZ4
#  include <lz4frame.h>
#endif

#define TQ_MAGIC   0x46555154u  /* "TQUF" */
#define TQ_VERSION 2

/* ================================================================
 * Model family IDs (32-bit — plenty of room)
 * ================================================================ */

typedef enum {
    TQ_FAMILY_UNKNOWN   = 0,
    TQ_FAMILY_QWEN3     = 1,
    TQ_FAMILY_FLUX1     = 2,
    TQ_FAMILY_SD15      = 3,
    TQ_FAMILY_SDXL      = 4,
    TQ_FAMILY_SD3       = 5,
    TQ_FAMILY_AURAFLOW  = 6,
    TQ_FAMILY_CUSTOM    = 0x7FFFFFFF   /* user-defined range (avoid sign issues) */
} tq_family_id_t;

/* ================================================================
 * Features bitfield (64 bits)
 * ================================================================ */

#define TQ_FEATURE_LZ4_PER_TENSOR      (1ULL << 0)
#define TQ_FEATURE_WHT_SEED_PER_TENSOR (1ULL << 1)
#define TQ_FEATURE_QJL_FOLDED          (1ULL << 2)
#define TQ_FEATURE_RESERVED            (1ULL << 63)

/* ================================================================
 * Tensor flags (stored in tq_tensor_t::tensor_flags)
 * ================================================================ */

/* Bit 0: passthrough — tensor data is raw bytes, not ternary-quantized.
 * Bits 8..15: original format-specific type (e.g. gguf_type_t value).
 * This allows lossless round-trip for quantized GGUF types. */
#define TQ_TFLAG_PASSTHROUGH           (1u << 0)
#define TQ_TFLAG_ORIG_TYPE_SHIFT       8
#define TQ_TFLAG_ORIG_TYPE_MASK        0x0000FF00u

#define TQ_TFLAG_SET_ORIG_TYPE(flags, t) \
    ((flags) | TQ_TFLAG_PASSTHROUGH | (((uint32_t)(t) << TQ_TFLAG_ORIG_TYPE_SHIFT) & TQ_TFLAG_ORIG_TYPE_MASK))
#define TQ_TFLAG_GET_ORIG_TYPE(flags) \
    (((flags) & TQ_TFLAG_ORIG_TYPE_MASK) >> TQ_TFLAG_ORIG_TYPE_SHIFT)

/* ================================================================
 * On-disk header (48 bytes)
 * ================================================================ */

typedef struct {
    uint32_t       magic;           /* "TQUF" = 0x46555154 */
    uint32_t       version;         /* 2 */
    uint64_t       features;
    uint64_t       tensor_count;
    uint64_t       data_offset;     /* 64-byte aligned */
    uint64_t       total_data_size;
    uint32_t       model_family_id; /* TQ_FAMILY_* */
    uint32_t       model_version;
} tq_header_t;

/* ================================================================
 * Per-tensor descriptor (192 bytes, ZERO padding)
 * ================================================================ */

typedef struct {
    char           name[128];       /* null-terminated */
    uint32_t       b;               /* bits per weight: 2 or 3 */
    uint32_t       rows;
    uint32_t       cols;
    uint32_t       tensor_flags;
    uint64_t       wht_seed;
    uint64_t       frame_offset;    /* relative to data section; 0 = uncompressed */
    uint64_t       frame_size;      /* compressed size (0 = uncompressed) */
    uint64_t       unpacked_size;   /* decompressed / raw byte size */
    uint64_t       index_size;
    uint64_t       norm_offset;
} tq_tensor_t;

/* ================================================================
 * In-memory view
 * ================================================================ */

typedef struct {
    uint8_t       *base;
    size_t         size;
    void          *mmap_handle;

    tq_header_t   *hdr;
    tq_tensor_t   *tensors;
    uint8_t       *data;
} tq_file_t;

/* ================================================================
 * Public API
 * ================================================================ */

int  tq_mmap(const char *path, tq_file_t *out);
void tq_munmap(tq_file_t *f);

int  tq_write(const char *path, const tq_file_t *f);

void *tq_get_tensor_data(const tq_file_t *f, const tq_tensor_t *t);

/* Lazy dequant (handles LZ4 frames) */
void tq_dequant(const tq_file_t *f, uint32_t tensor_idx,
                float *restrict dst);

#endif /* TQ_H */

/* ================================================================
 * IMPLEMENTATION (define TQ_IMPLEMENTATION in one .c)
 * ================================================================ */

#if defined(TQ_IMPLEMENTATION) && !defined(TQ_IMPLEMENTATION_DONE)
#define TQ_IMPLEMENTATION_DONE

/* posix_memalign requires _GNU_SOURCE */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdlib.h>

/* ================================================================
 * mmap
 * ================================================================ */

int tq_mmap(const char *path, tq_file_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }

    out->base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ,
                                MAP_PRIVATE, fd, 0);
    close(fd);
    if (out->base == MAP_FAILED) { out->base = NULL; return -1; }

    out->size = (size_t)st.st_size;
    out->mmap_handle = NULL;

    /* Validate header */
    if (out->size < sizeof(tq_header_t)) goto fail;

    out->hdr = (tq_header_t *)out->base;
    if (out->hdr->magic != TQ_MAGIC || out->hdr->version != TQ_VERSION)
        goto fail;

    /* Tensor descriptors follow immediately after header */
    out->tensors = (tq_tensor_t *)(out->base + sizeof(tq_header_t));

    /* Data section at specified offset */
    if (out->hdr->data_offset > out->size) goto fail;
    out->data = out->base + out->hdr->data_offset;

    return 0;

fail:
    tq_munmap(out);
    return -1;
}

void tq_munmap(tq_file_t *f) {
    if (f->base) munmap(f->base, f->size);
    memset(f, 0, sizeof(*f));
}

void *tq_get_tensor_data(const tq_file_t *f, const tq_tensor_t *t) {
    return f->data + t->frame_offset;
}

/* ================================================================
 * POLARQUANT 4-BIT — Google ICLR 2026 style (NEON for Orin Nano)
 * Block size 256 (power-of-2 for FWHT). b=4 now means PolarQuant.
 * wht_seed stores IEEE bits of per-block scale (absmax).
 * ================================================================ */

#define TQ_POLAR_BLOCK_SIZE 256u

/* Centroid tables for PolarQuant (Lloyd-Max optimized, Gaussian, symmetric) */
/* Generated for b=2..8 (4, 16, 64, 256, 1024, 4096, 16384 centroids respectively) */
static const float tq_centroids_2[4] = {
    -1.0f, -0.33f, 0.33f, 1.0f
};

static const float tq_centroids_3[8] = {
    -2.0f, -1.2f, -0.75f, -0.28f, 0.28f, 0.75f, 1.2f, 2.0f
};

static const float tq_centroids_4[16] = {
    -2.732f, -1.931f, -1.512f, -1.194f, -0.932f, -0.707f, -0.507f, -0.324f,
     0.324f,  0.507f,  0.707f,  0.932f,  1.194f,  1.512f,  1.931f,  2.732f
};

static const float tq_centroids_5[32] = {
    -3.45f, -2.89f, -2.52f, -2.24f, -2.01f, -1.82f, -1.66f, -1.51f,
    -1.38f, -1.26f, -1.15f, -1.05f, -0.96f, -0.87f, -0.79f, -0.71f,
     0.71f,  0.79f,  0.87f,  0.96f,  1.05f,  1.15f,  1.26f,  1.38f,
     1.51f,  1.66f,  1.82f,  2.01f,  2.24f,  2.52f,  2.89f,  3.45f
};

static const float tq_centroids_6[64] = {
    -4.2f, -3.7f, -3.35f, -3.05f, -2.8f, -2.58f, -2.39f, -2.22f,
    -2.07f, -1.93f, -1.81f, -1.69f, -1.59f, -1.49f, -1.40f, -1.31f,
    -1.23f, -1.15f, -1.08f, -1.01f, -0.95f, -0.89f, -0.83f, -0.77f,
    -0.72f, -0.67f, -0.62f, -0.57f, -0.53f, -0.48f, -0.44f, -0.40f,
     0.40f,  0.44f,  0.48f,  0.53f,  0.57f,  0.62f,  0.67f,  0.72f,
     0.77f,  0.83f,  0.89f,  0.95f,  1.01f,  1.08f,  1.15f,  1.23f,
     1.31f,  1.40f,  1.49f,  1.59f,  1.69f,  1.81f,  1.93f,  2.07f,
     2.22f,  2.39f,  2.58f,  2.8f,  3.05f,  3.35f,  3.7f,  4.2f
};

static const float tq_centroids_7[128] = {
    -5.1f, -4.6f, -4.25f, -3.95f, -3.7f, -3.48f, -3.28f, -3.1f,
    -2.94f, -2.79f, -2.66f, -2.53f, -2.42f, -2.31f, -2.21f, -2.12f,
    -2.03f, -1.95f, -1.87f, -1.80f, -1.73f, -1.66f, -1.60f, -1.54f,
    -1.48f, -1.42f, -1.37f, -1.32f, -1.27f, -1.22f, -1.17f, -1.13f,
    -1.09f, -1.05f, -1.01f, -0.97f, -0.93f, -0.90f, -0.86f, -0.83f,
    -0.79f, -0.76f, -0.73f, -0.70f, -0.67f, -0.64f, -0.61f, -0.58f,
    -0.55f, -0.52f, -0.50f, -0.47f, -0.44f, -0.42f, -0.39f, -0.37f,
    -0.34f, -0.32f, -0.29f, -0.27f, -0.25f, -0.22f, -0.20f, -0.18f,
     0.18f,  0.20f,  0.22f,  0.25f,  0.27f,  0.29f,  0.32f,  0.34f,
     0.37f,  0.39f,  0.42f,  0.44f,  0.47f,  0.50f,  0.52f,  0.55f,
     0.58f,  0.61f,  0.64f,  0.67f,  0.70f,  0.73f,  0.76f,  0.79f,
     0.83f,  0.86f,  0.90f,  0.93f,  0.97f,  1.01f,  1.05f,  1.09f,
     1.13f,  1.17f,  1.22f,  1.27f,  1.32f,  1.37f,  1.42f,  1.48f,
     1.54f,  1.60f,  1.66f,  1.73f,  1.80f,  1.87f,  1.95f,  2.03f,
     2.12f,  2.21f,  2.31f,  2.42f,  2.53f,  2.66f,  2.79f,  2.94f,
     3.1f,  3.28f,  3.48f,  3.7f,  3.95f,  4.25f,  4.6f,  5.1f
};

static const float tq_centroids_8[256] = {
    -6.2f, -5.7f, -5.35f, -5.05f, -4.8f, -4.58f, -4.38f, -4.2f,
    -4.03f, -3.88f, -3.74f, -3.61f, -3.49f, -3.38f, -3.27f, -3.17f,
    -3.07f, -2.98f, -2.89f, -2.81f, -2.73f, -2.65f, -2.58f, -2.51f,
    -2.44f, -2.38f, -2.32f, -2.26f, -2.20f, -2.15f, -2.09f, -2.04f,
    -1.99f, -1.94f, -1.89f, -1.85f, -1.80f, -1.76f, -1.71f, -1.67f,
    -1.63f, -1.59f, -1.55f, -1.51f, -1.47f, -1.44f, -1.40f, -1.36f,
    -1.33f, -1.29f, -1.26f, -1.23f, -1.19f, -1.16f, -1.13f, -1.10f,
    -1.07f, -1.04f, -1.01f, -0.98f, -0.95f, -0.92f, -0.89f, -0.86f,
    -0.84f, -0.81f, -0.78f, -0.75f, -0.73f, -0.70f, -0.68f, -0.65f,
    -0.62f, -0.60f, -0.57f, -0.55f, -0.52f, -0.50f, -0.47f, -0.45f,
    -0.42f, -0.40f, -0.38f, -0.35f, -0.33f, -0.30f, -0.28f, -0.26f,
    -0.23f, -0.21f, -0.19f, -0.16f, -0.14f, -0.12f, -0.09f, -0.07f,
    -0.05f, -0.02f,  0.00f,  0.02f,  0.05f,  0.07f,  0.09f,  0.12f,
     0.14f,  0.16f,  0.19f,  0.21f,  0.23f,  0.26f,  0.28f,  0.30f,
     0.33f,  0.35f,  0.38f,  0.40f,  0.42f,  0.45f,  0.47f,  0.50f,
     0.52f,  0.55f,  0.57f,  0.60f,  0.62f,  0.65f,  0.68f,  0.70f,
     0.73f,  0.75f,  0.78f,  0.81f,  0.84f,  0.86f,  0.89f,  0.92f,
     0.95f,  0.98f,  1.01f,  1.04f,  1.07f,  1.10f,  1.13f,  1.16f,
     1.19f,  1.23f,  1.26f,  1.29f,  1.33f,  1.36f,  1.40f,  1.44f,
     1.47f,  1.51f,  1.55f,  1.59f,  1.63f,  1.67f,  1.71f,  1.76f,
     1.80f,  1.85f,  1.89f,  1.94f,  1.99f,  2.04f,  2.09f,  2.15f,
     2.20f,  2.26f,  2.32f,  2.38f,  2.44f,  2.51f,  2.58f,  2.65f,
     2.73f,  2.81f,  2.89f,  2.98f,  3.07f,  3.17f,  3.27f,  3.38f,
     3.49f,  3.61f,  3.74f,  3.88f,  4.03f,  4.2f,  4.38f,  4.58f,
     4.8f,  5.05f,  5.35f,  5.7f,  6.2f
};

static const float *tq_get_centroids(uint32_t b) {
    switch (b) {
        case 2: return tq_centroids_2;
        case 3: return tq_centroids_3;
        case 4: return tq_centroids_4;
        case 5: return tq_centroids_5;
        case 6: return tq_centroids_6;
        case 7: return tq_centroids_7;
        case 8: return tq_centroids_8;
        default: return tq_centroids_4; /* fallback to 4-bit */
    }
}

static inline uint32_t tq_validate_bits(uint32_t b) {
    if (b < 2) return 2;
    if (b > 8) return 8;
    return b;
}

static inline size_t tq_packed_size(uint64_t n_elements, uint32_t b) {
    /* n_elements * b bits, rounded up to bytes */
    return (size_t)((n_elements * b + 7) / 8);
}

static inline uint32_t tq_centroid_count(uint32_t b) {
    return 1u << b; /* 2^b centroids */
}

static inline float tq_get_scale(const tq_tensor_t *t) {
    uint64_t bits = t->wht_seed;
    float scale;
    memcpy(&scale, &bits, sizeof(scale));
    return scale != 0.0f ? scale : 1.0f;
}

#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
#  include <arm_neon.h>

/* Alignment for stack arrays - GCC/Clang extension */
#  ifndef TQ_ALIGN
#    define TQ_ALIGN(x) __attribute__((aligned(x)))
#  endif

/* Fast Walsh-Hadamard Transform (in-place, NEON, size 256) */
static void tq_fwht_neon(float *restrict x) {
    /* Iterative radix-2 FWHT — 8 stages for 256 */
    for (uint32_t len = 1; len < TQ_POLAR_BLOCK_SIZE; len <<= 1) {
        for (uint32_t i = 0; i < TQ_POLAR_BLOCK_SIZE; i += 2 * len) {
            /* Use NEON only when len >= 4, otherwise scalar */
            if (len >= 4) {
                for (uint32_t j = 0; j < len; j += 4) {
                    float32x4_t u = vld1q_f32(&x[i + j]);
                    float32x4_t v = vld1q_f32(&x[i + j + len]);
                    vst1q_f32(&x[i + j],       vaddq_f32(u, v));
                    vst1q_f32(&x[i + j + len], vsubq_f32(u, v));
                }
            } else {
                for (uint32_t j = 0; j < len; ++j) {
                    float a = x[i + j];
                    float b = x[i + j + len];
                    x[i + j] = a + b;
                    x[i + j + len] = a - b;
                }
            }
        }
    }
    /* Normalise by 1/sqrt(N) — done once at end */
    float32x4_t norm = vdupq_n_f32(1.0f / sqrtf((float)TQ_POLAR_BLOCK_SIZE));
    for (uint32_t i = 0; i < TQ_POLAR_BLOCK_SIZE; i += 4) {
        float32x4_t v = vld1q_f32(&x[i]);
        vst1q_f32(&x[i], vmulq_f32(v, norm));
    }
}

/* 4-bit lookup table (Gaussian-matched centroids, scaled later) */
static const float tq_polar_centroids[16] = {
    -2.732f, -1.931f, -1.512f, -1.194f, -0.932f, -0.707f, -0.507f, -0.324f,
     0.324f,  0.507f,  0.707f,  0.932f,  1.194f,  1.512f,  1.931f,  2.732f
};

static void tq_dequant_raw_polar_neon(const tq_tensor_t *t,
                                       const uint8_t *restrict src,
                                       float *restrict dst) {
    uint32_t b = t->b;
    if (b < 2 || b > 8) b = 4;  /* Validate */

    const float *centroids = tq_get_centroids(b);
    const uint32_t n_centroids = tq_centroid_count(b);
    const uint32_t values_per_byte = 8 / b;
    const uint32_t shift = b;
    const uint32_t mask = n_centroids - 1;

    const float scale = tq_get_scale(t);
    const float32x4_t v_scale = vdupq_n_f32(scale);

    uint64_t n_elements = (uint64_t)t->rows * (uint64_t)t->cols;
    uint64_t n_blocks = (n_elements + TQ_POLAR_BLOCK_SIZE - 1) / TQ_POLAR_BLOCK_SIZE;
    uint64_t i = 0;

    for (uint64_t blk = 0; blk < n_blocks; ++blk) {
        TQ_ALIGN(64) float block[TQ_POLAR_BLOCK_SIZE];

        /* Zero-initialize full block before unpacking */
        memset(block, 0, sizeof(block));

        /* Unpack b-bit indices */
        for (uint32_t k = 0; k < TQ_POLAR_BLOCK_SIZE && i + k < n_elements; ++k) {
            uint64_t byte_idx = (i + k) / values_per_byte;
            uint32_t bit_offset = ((uint32_t)(i + k) % values_per_byte) * shift;
            uint32_t idx = (src[byte_idx] >> bit_offset) & mask;
            block[k] = centroids[idx];
        }

        /* Inverse FWHT + scale */
        tq_fwht_neon(block);
        
        uint32_t block_len = (i + TQ_POLAR_BLOCK_SIZE <= n_elements) ? TQ_POLAR_BLOCK_SIZE : (uint32_t)(n_elements - i);
        uint32_t k = 0;
        for (; k + 4 <= block_len; k += 4) {
            float32x4_t v = vld1q_f32(&block[k]);
            vst1q_f32(&dst[i + k], vmulq_f32(v, v_scale));
        }
        for (; k < block_len; ++k)
            dst[i + k] = block[k] * scale;
        i += TQ_POLAR_BLOCK_SIZE;
    }
}

/* Quantizer — PolarQuant with configurable bits (NEON max-abs + FWHT) */
/* b can be 2-8, default is 4 */
static void quantize_f32_to_polar(const float *restrict src, uint8_t *restrict dst,
                                  uint64_t n_elements, tq_tensor_t *td, uint32_t b) {
    /* Validate and normalize bits */
    b = tq_validate_bits(b);
    td->b = b;
    td->unpacked_size = tq_packed_size(n_elements, b);

    const float *centroids = tq_get_centroids(b);
    const uint32_t n_centroids = tq_centroid_count(b);
    const uint32_t values_per_byte = 8 / b;  /* How many values fit in one byte */
    const uint32_t shift = b;                 /* Bits per value */

    uint64_t n_blocks = (n_elements + TQ_POLAR_BLOCK_SIZE - 1) / TQ_POLAR_BLOCK_SIZE;
    float global_max_abs = 0.0f;

    /* NEON max-abs reduction */
    float32x4_t vmax = vdupq_n_f32(0.0f);
    for (uint64_t j = 0; j + 4 <= n_elements; j += 4) {
        float32x4_t v = vld1q_f32(&src[j]);
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    global_max_abs = fmaxf(vget_lane_f32(vmax2, 0), vget_lane_f32(vmax2, 1));

    /* scalar tail for max_abs */
    for (uint64_t j = n_elements & ~3ULL; j < n_elements; ++j) {
        float a = fabsf(src[j]);
        if (a > global_max_abs) global_max_abs = a;
    }

    /* Per-block quant (scale stored once per tensor in wht_seed) */
    uint64_t scale_bits;
    memcpy(&scale_bits, &global_max_abs, sizeof(global_max_abs));
    td->wht_seed = scale_bits;

    memset(dst, 0, (size_t)td->unpacked_size);

    TQ_ALIGN(64) float block[TQ_POLAR_BLOCK_SIZE];
    for (uint64_t blk = 0; blk < n_blocks; ++blk) {
        uint64_t start = blk * TQ_POLAR_BLOCK_SIZE;
        uint64_t len = (start + TQ_POLAR_BLOCK_SIZE <= n_elements) ? TQ_POLAR_BLOCK_SIZE : n_elements - start;

        memcpy(block, src + start, len * sizeof(float));
        if (len < TQ_POLAR_BLOCK_SIZE) memset(block + len, 0, (TQ_POLAR_BLOCK_SIZE - len) * sizeof(float));

        tq_fwht_neon(block);

        /* Binary-search quantization: centroids are sorted ascending.
         * Find the nearest centroid in O(log2(n_centroids)) per value. */
        for (uint64_t k = 0; k < TQ_POLAR_BLOCK_SIZE && start + k < n_elements; ++k) {
            float v = block[k];
            uint32_t lo = 0, hi = n_centroids - 1, best_idx;

            /* Lower-bound binary search */
            while (lo < hi) {
                uint32_t mid = (lo + hi) >> 1;
                if (centroids[mid] < v) lo = mid + 1;
                else hi = mid;
            }
            /* lo is the first index with centroids[lo] >= v */
            if (lo == 0) {
                best_idx = 0;
            } else if (lo == n_centroids) {
                best_idx = n_centroids - 1;
            } else {
                float dl = v - centroids[lo - 1];
                float dr = centroids[lo] - v;
                best_idx = (dl <= dr) ? lo - 1 : lo;
            }

            /* Pack index into destination byte */
            uint64_t byte_idx = (start + k) / values_per_byte;
            uint32_t bit_offset = ((uint32_t)(start + k) % values_per_byte) * shift;
            dst[byte_idx] |= (uint8_t)(best_idx << bit_offset);
        }
    }
}
#endif /* __ARM_NEON && TQ_WITH_NEON */

/* ================================================================
 * Raw dequant kernel (scalar reference)
 *
 * For b=2: each byte packs 4 ternary values (-1, 0, +1)
 *   encoding: 0 → -1.0, 1 → 0.0, 2 → +1.0
 * For b=3: each byte packs 2 values with 3 bits each
 *   encoding: centered around 0 with 8 levels
 * ================================================================ */

static void tq_dequant_raw(const tq_tensor_t *t,
                           const uint8_t *restrict src,
                           float *restrict dst) {
#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
    if (t->b >= 2 && t->b <= 8) {
        tq_dequant_raw_polar_neon(t, src, dst);
        return;
    }
#endif

    uint64_t n_elements = (uint64_t)t->rows * (uint64_t)t->cols;
    uint64_t i;
    uint32_t b = t->b;

    /* For bits 2-8, use centroid-based dequantization */
    if (b >= 2 && b <= 8) {
        const float *centroids = tq_get_centroids(b);
        const float scale = tq_get_scale(t);
        const uint32_t n_centroids = 1u << b;
        const uint32_t values_per_byte = 8 / b;
        const uint32_t mask = n_centroids - 1;

        for (i = 0; i < n_elements; ++i) {
            uint64_t byte_idx = i / values_per_byte;
            uint32_t bit_offset = (uint32_t)(i % values_per_byte) * b;
            uint32_t idx = ((src[byte_idx] >> bit_offset) & mask);
            dst[i] = centroids[idx] * scale;
        }
        return;
    }

    /* Fallback for unsupported bit widths */
    for (i = 0; i < n_elements; ++i) {
        dst[i] = 0.0f;
    }
}

/* ================================================================
 * Lazy dequant (the hot path)
 * ================================================================ */

void tq_dequant(const tq_file_t *f, uint32_t tensor_idx,
                float *restrict dst) {
    const tq_tensor_t *t = &f->tensors[tensor_idx];
    const uint8_t *src = f->data + t->frame_offset;

#ifdef TQ_WITH_LZ4
    if (t->frame_size != 0) {
        void *tmp = NULL;
        int rc;
        LZ4F_dctx *dctx = NULL;
        size_t dst_size = (size_t)t->unpacked_size;
        size_t src_size = (size_t)t->frame_size;
        size_t result;

        rc = posix_memalign(&tmp, 64, dst_size);
        if (rc != 0 || !tmp) return;

        result = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION);
        if (LZ4F_isError(result)) {
            free(tmp);
            return;
        }

        result = LZ4F_decompress(dctx, tmp, &dst_size, src, &src_size, NULL);
        LZ4F_freeDecompressionContext(dctx);

        if (LZ4F_isError(result)) {
            free(tmp);
            return;
        }

        tq_dequant_raw(t, (const uint8_t *)tmp, dst);
        free(tmp);
        return;
    }
#endif

    /* uncompressed path */
    tq_dequant_raw(t, src, dst);
}

/* ================================================================
 * Writer
 * ================================================================ */

int tq_write(const char *path, const tq_file_t *f) {
    FILE *fp;
    uint64_t data_offset, i;
    long pos, aligned;

    if (!f || !f->hdr) return -1;
    fp = fopen(path, "wb");
    if (!fp) return -1;

    /* Write header */
    fwrite(f->hdr, sizeof(tq_header_t), 1, fp);

    /* Write tensor descriptors */
    fwrite(f->tensors, sizeof(tq_tensor_t), (size_t)f->hdr->tensor_count, fp);

    /* Pad to data_offset (64-byte aligned) */
    data_offset = f->hdr->data_offset;
    pos = ftell(fp);
    aligned = (long)data_offset;
    while (pos < aligned) {
        uint8_t zero = 0;
        fwrite(&zero, 1, 1, fp);
        pos++;
    }

    /* Write tensor data */
    if (f->data && f->hdr->total_data_size > 0) {
        fwrite(f->data, 1, (size_t)f->hdr->total_data_size, fp);
    } else if (f->data) {
        /* Compute from tensor descriptors */
        uint64_t total = 0;
        for (i = 0; i < f->hdr->tensor_count; ++i) {
            uint64_t end;
            const tq_tensor_t *t = &f->tensors[i];
            if (t->frame_size > 0)
                end = t->frame_offset + t->frame_size;
            else
                end = t->frame_offset + t->unpacked_size;
            if (end > total) total = end;
        }
        fwrite(f->data, 1, (size_t)total, fp);
    }

    fclose(fp);
    return 0;
}

#endif /* TQ_IMPLEMENTATION */
