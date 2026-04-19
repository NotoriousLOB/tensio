/* convert.h — any-to-any model format converter
 * Strict C99, zero UB, header-only
 * Uses safetensors.h, gguf.h, tq.h
 */

#ifndef CONVERT_H
#define CONVERT_H

/* Format headers are expected to be included before this file
 * (either directly or via tensio.h). We only need their types. */
#ifndef SAFETENSORS_H
#  include "safetensors.h"
#endif
#ifndef GGUF_H
#  include "gguf.h"
#endif
#ifndef TQ_H
#  include "tq.h"
#endif

/* ================================================================
 * High-level any-to-any conversion
 * ================================================================ */

/* Conversion options */
typedef struct {
    int use_lz4;           /* 1 = enable per-tensor LZ4 compression for TQ output */
    uint32_t bits_per_weight; /* 2-8, default 4 (PolarQuant bits per weight) */
} convert_opts_t;

/* Returns 0 on success */
int convert_any_to_any(const char *input_path, const char *output_path);
int convert_any_to_any_opts(const char *input_path, const char *output_path,
                            const convert_opts_t *opts);

/* Explicit converters */
int convert_safetensors_to_gguf(const char *st_path, const char *gguf_path);
int convert_safetensors_to_tq(const char *st_path, const char *tq_path);
int convert_safetensors_to_tq_opts(const char *st_path, const char *tq_path,
                                   const convert_opts_t *opts);

int convert_gguf_to_safetensors(const char *gguf_path, const char *st_path);
int convert_gguf_to_tq(const char *gguf_path, const char *tq_path);
int convert_gguf_to_tq_opts(const char *gguf_path, const char *tq_path,
                            const convert_opts_t *opts);

int convert_tq_to_safetensors(const char *tq_path, const char *st_path);
int convert_tq_to_gguf(const char *tq_path, const char *gguf_path);

#endif /* CONVERT_H */

/* ================================================================
 * IMPLEMENTATION (define CONVERT_IMPLEMENTATION in one .c file)
 * ================================================================ */

#if defined(CONVERT_IMPLEMENTATION) && !defined(CONVERT_IMPLEMENTATION_DONE)
#define CONVERT_IMPLEMENTATION_DONE

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static int detect_format(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint32_t magic = 0;
    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return -1; }
    fclose(f);

    if (magic == 0x46554747u) return 1;        /* GGUF */
    if (magic == 0x46555154u) return 2;        /* TQ */
    /* Safetensors has no magic, but starts with 8-byte length */
    return 0;   /* assume Safetensors */
}

/* ================================================================
 * Type mapping: Safetensors ↔ GGUF
 *
 * Only lossless mappings for standard float/int types.
 * Quantized GGUF types have no Safetensors equivalent.
 * ================================================================ */

static gguf_type_t st_dtype_to_gguf(st_dtype_t dt) {
    switch (dt) {
        case ST_F32:  return GGUF_TYPE_F32;
        case ST_F16:  return GGUF_TYPE_F16;
        case ST_BF16: return GGUF_TYPE_BF16;
        case ST_F64:  return GGUF_TYPE_F64;
        case ST_I8:   return GGUF_TYPE_I8;
        case ST_I16:  return GGUF_TYPE_I16;
        case ST_I32:  return GGUF_TYPE_I32;
        case ST_I64:  return GGUF_TYPE_I64;
        default:      return GGUF_TYPE_F32; /* fallback */
    }
}

static st_dtype_t gguf_type_to_st(gguf_type_t gt) {
    switch (gt) {
        case GGUF_TYPE_F32:  return ST_F32;
        case GGUF_TYPE_F16:  return ST_F16;
        case GGUF_TYPE_BF16: return ST_BF16;
        case GGUF_TYPE_F64:  return ST_F64;
        case GGUF_TYPE_I8:   return ST_I8;
        case GGUF_TYPE_I16:  return ST_I16;
        case GGUF_TYPE_I32:  return ST_I32;
        case GGUF_TYPE_I64:  return ST_I64;
        default:             return ST_F32; /* GGUF quantized types (Q4_0, Q4_K, etc.)
                                             * have no ST equivalent — callers must
                                             * dequantize first or use passthrough. */
    }
}

/* ================================================================
 * Safetensors → GGUF
 *
 * Maps each st_tensor_t to a gguf_tensor_t, copies raw data.
 * ================================================================ */

int convert_safetensors_to_gguf(const char *st_path, const char *gguf_path) {
    st_mmap_t sm;
    st_file_t src;
    gguf_file_t dst;
    uint32_t i;
    int rc;
    uint64_t data_offset;
    uint8_t *data_buf;

    if (st_mmap(st_path, &sm) != 0) return -1;
    if (st_parse(&sm, &src) != 0) { st_munmap(&sm); return -1; }

    memset(&dst, 0, sizeof(dst));
    dst.magic = GGUF_MAGIC;
    dst.version = GGUF_VERSION;
    dst.tensor_count = src.num_tensors;
    dst.metadata_count = 0;
    dst.metadata = NULL;

    dst.tensors = (gguf_tensor_t *)calloc(src.num_tensors, sizeof(gguf_tensor_t));
    if (!dst.tensors && src.num_tensors > 0) {
        st_free(&src); st_munmap(&sm); return -1;
    }

    data_offset = 0;
    for (i = 0; i < src.num_tensors; ++i) {
        const st_tensor_t *st = &src.tensors[i];
        gguf_tensor_t *gt = &dst.tensors[i];
        uint32_t d;

        gt->name = (char *)malloc(strlen(st->name) + 1);
        if (gt->name) strcpy(gt->name, st->name);

        gt->type = st_dtype_to_gguf(st->dtype);
        memset(gt->ne, 0, sizeof(gt->ne));
        if (st->ndim <= 4) {
            gt->n_dims = st->ndim;
            for (d = 0; d < st->ndim; ++d)
                gt->ne[d] = st->shape[d];
        } else {
            /* GGUF v3 supports max 4 dims — flatten to 1D */
            uint64_t total = 1;
            for (d = 0; d < st->ndim; ++d) total *= st->shape[d];
            gt->n_dims = 1;
            gt->ne[0] = total;
        }

        /* GGUF spec: tensor offsets relative to data section must be 32-byte aligned */
        data_offset = (data_offset + 31) & ~(uint64_t)31;
        gt->offset = data_offset;
        gt->size = st->size;
        data_offset += st->size;
    }

    /* Build contiguous data buffer (size includes alignment padding) */
    data_buf = (uint8_t *)calloc(1, (size_t)data_offset);
    if (!data_buf && data_offset > 0) {
        for (i = 0; i < src.num_tensors; ++i) free(dst.tensors[i].name);
        free(dst.tensors);
        st_free(&src); st_munmap(&sm); return -1;
    }

    for (i = 0; i < src.num_tensors; ++i) {
        const st_tensor_t *st = &src.tensors[i];
        void *src_data = st_get_tensor_data(&src, st);
        memcpy(data_buf + dst.tensors[i].offset, src_data, (size_t)st->size);
    }
    dst.data = data_buf;

    rc = gguf_write(gguf_path, &dst);

    free(data_buf);
    for (i = 0; i < src.num_tensors; ++i) free(dst.tensors[i].name);
    free(dst.tensors);
    st_free(&src);
    st_munmap(&sm);
    return rc;
}

/* ================================================================
 * Safetensors → TQ
 *
 * Only meaningful for F32 data that can be ternary-quantized.
 * Uses simple threshold-based b=2 quantization:
 *   val < -0.33 → -1, val > 0.33 → +1, else → 0
 * ================================================================ */

static int convert_opts_lz4(const convert_opts_t *opts) {
#ifdef TQ_WITH_LZ4
    return opts && opts->use_lz4;
#else
    (void)opts;
    return 0;
#endif
}

static uint32_t convert_opts_bits(const convert_opts_t *opts) {
    /* Default to 4 bits per weight if not specified */
    if (!opts || opts->bits_per_weight == 0) return 4;
    /* Clamp to valid range [2, 8] */
    if (opts->bits_per_weight < 2) return 2;
    if (opts->bits_per_weight > 8) return 8;
    return opts->bits_per_weight;
}

static void quantize_f32_to_ternary(const float *src, uint8_t *dst,
                                    uint64_t n_elements) {
    uint64_t i;
    memset(dst, 0, (size_t)((n_elements + 3) / 4));
    for (i = 0; i < n_elements; ++i) {
        uint8_t val;
        uint64_t byte_idx = i / 4;
        uint32_t shift = (uint32_t)((i % 4) * 2);
        if (src[i] < -0.33f) val = 0;      /* -1 */
        else if (src[i] > 0.33f) val = 2;  /* +1 */
        else val = 1;                        /*  0 */
        dst[byte_idx] |= (uint8_t)(val << shift);
    }
}

int convert_safetensors_to_tq_opts(const char *st_path, const char *tq_path,
                                   const convert_opts_t *opts) {
    st_mmap_t sm;
    st_file_t src;
    tq_header_t hdr;
    tq_tensor_t *descs = NULL;
    FILE *fp = NULL;
    uint32_t i;
    int rc = -1;
    int lz4 = convert_opts_lz4(opts);
    long desc_start, aligned;
    uint64_t data_offset = 0;
    uint32_t bits = convert_opts_bits(opts);

    if (st_mmap(st_path, &sm) != 0) return -1;
    if (st_parse(&sm, &src) != 0) { st_munmap(&sm); return -1; }

    descs = (tq_tensor_t *)calloc(src.num_tensors, sizeof(tq_tensor_t));
    if (!descs && src.num_tensors > 0) goto done;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQ_MAGIC;
    hdr.version = TQ_VERSION;
    hdr.tensor_count = src.num_tensors;
    hdr.model_family_id = TQ_FAMILY_UNKNOWN;
    if (lz4) hdr.features |= TQ_FEATURE_LZ4_PER_TENSOR;

    /* Compute data section start: header + all descriptors, 64-byte aligned */
    desc_start = (long)(sizeof(tq_header_t) + (size_t)src.num_tensors * sizeof(tq_tensor_t));
    aligned = (desc_start + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;

    fp = fopen(tq_path, "wb");
    if (!fp) goto done;

    /* Write placeholder header and descriptors; we'll seek back to fix them */
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(descs, sizeof(tq_tensor_t), src.num_tensors, fp);
    {
        long pos = ftell(fp);
        while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }
    }

    /* Streaming pass: quantize one tensor at a time, write, free immediately */
    for (i = 0; i < src.num_tensors; ++i) {
        const st_tensor_t *st = &src.tensors[i];
        tq_tensor_t *td = &descs[i];
        uint64_t n_elements;
        uint8_t *buf = NULL;
        uint32_t d;

        strncpy(td->name, st->name, sizeof(td->name) - 1);
        n_elements = 1;
        for (d = 0; d < st->ndim; ++d) n_elements *= st->shape[d];
        td->rows = (uint32_t)n_elements;
        td->cols = 1;
        td->frame_offset = data_offset;

        if (st->dtype == ST_F32 || st->dtype == ST_BF16) {
            float *tmp = NULL;
            const float *fdata;

            if (st->dtype == ST_BF16) {
                const uint16_t *src16 =
                    (const uint16_t *)st_get_tensor_data(&src, st);
                tmp = (float *)malloc(n_elements * sizeof(float));
                if (tmp) {
                    uint64_t j;
                    for (j = 0; j < n_elements; ++j)
                        tmp[j] = bf16_to_f32(src16[j]);
                }
                fdata = tmp;
            } else {
                fdata = (const float *)st_get_tensor_data(&src, st);
            }

            if (fdata) {
#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
                buf = (uint8_t *)malloc(tq_packed_size(n_elements, bits));
                if (buf) quantize_f32_to_polar(fdata, buf, n_elements, td, bits);
#else
                td->b = 2;
                td->unpacked_size = (n_elements + 3) / 4;
                buf = (uint8_t *)calloc(1, (size_t)td->unpacked_size);
                if (buf) quantize_f32_to_ternary(fdata, buf, n_elements);
#endif
            }
            free(tmp);
        } else {
            td->b = 0;
            td->tensor_flags = TQ_TFLAG_SET_ORIG_TYPE(0, (uint32_t)st->dtype);
            td->unpacked_size = st->size;
            buf = (uint8_t *)malloc((size_t)st->size);
            if (buf) memcpy(buf, st_get_tensor_data(&src, st), (size_t)st->size);
        }

        if (!buf) goto done;

#ifdef TQ_WITH_LZ4
        if (lz4 && td->unpacked_size > 0) {
            LZ4F_preferences_t prefs;
            size_t bound, csize;
            uint8_t *cbuf;

            memset(&prefs, 0, sizeof(prefs));
            prefs.frameInfo.contentSize = td->unpacked_size;
            bound = LZ4F_compressFrameBound(td->unpacked_size, &prefs);
            cbuf = (uint8_t *)malloc(bound);
            if (cbuf) {
                csize = LZ4F_compressFrame(cbuf, bound, buf, td->unpacked_size, &prefs);
                if (!LZ4F_isError(csize) && csize < td->unpacked_size) {
                    td->frame_size = csize;
                    fwrite(cbuf, 1, csize, fp);
                    data_offset += csize;
                    free(cbuf);
                    free(buf);
                    buf = NULL;
                } else {
                    free(cbuf);
                    /* fall through to uncompressed write below */
                }
            }
        }
#endif
        if (buf) {
            td->frame_size = 0;
            fwrite(buf, 1, (size_t)td->unpacked_size, fp);
            data_offset += td->unpacked_size;
            free(buf);
        }
    }

    hdr.total_data_size = data_offset;

    /* Seek back and rewrite header + descriptors with final offsets */
    if (fseek(fp, 0, SEEK_SET) != 0) goto done;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(descs, sizeof(tq_tensor_t), src.num_tensors, fp);
    rc = 0;

done:
    if (fp) fclose(fp);
    if (rc != 0 && tq_path) remove(tq_path);
    free(descs);
    st_free(&src);
    st_munmap(&sm);
    return rc;
}

int convert_safetensors_to_tq(const char *st_path, const char *tq_path) {
    return convert_safetensors_to_tq_opts(st_path, tq_path, NULL);
}

/* ================================================================
 * GGUF → Safetensors
 *
 * Maps each gguf_tensor_t to st_tensor_t, copies raw data.
 * Quantized GGUF types are mapped to F32 dtype but data is
 * copied as raw bytes (consumer must dequantize).
 * ================================================================ */

int convert_gguf_to_safetensors(const char *gguf_path, const char *st_path) {
    gguf_mmap_t gm;
    gguf_file_t src;
    st_file_t dst;
    uint64_t i;
    uint64_t data_offset;
    uint8_t *data_buf;
    int rc;

    if (gguf_mmap(gguf_path, &gm) != 0) return -1;
    if (gguf_parse(&gm, &src) != 0) { gguf_munmap(&gm); return -1; }

    memset(&dst, 0, sizeof(dst));
    dst.num_tensors = (uint32_t)src.tensor_count;
    dst.tensors = (st_tensor_t *)calloc((size_t)src.tensor_count, sizeof(st_tensor_t));
    if (!dst.tensors && src.tensor_count > 0) {
        gguf_free(&src); gguf_munmap(&gm); return -1;
    }

    data_offset = 0;
    for (i = 0; i < src.tensor_count; ++i) {
        const gguf_tensor_t *gt = &src.tensors[i];
        st_tensor_t *st = &dst.tensors[i];
        uint32_t d;

        st->name = (char *)malloc(strlen(gt->name) + 1);
        if (st->name) strcpy(st->name, gt->name);

        st->dtype = gguf_type_to_st(gt->type);
        st->ndim = gt->n_dims;
        if (st->ndim > 8) st->ndim = 8;
        for (d = 0; d < st->ndim; ++d)
            st->shape[d] = gt->ne[d];

        st->offset = data_offset;
        st->size = gt->size;
        data_offset += gt->size;
    }

    dst.data_size = data_offset;

    /* Build contiguous data buffer */
    data_buf = (uint8_t *)malloc((size_t)data_offset);
    if (!data_buf && data_offset > 0) {
        st_free(&dst); gguf_free(&src); gguf_munmap(&gm); return -1;
    }

    for (i = 0; i < src.tensor_count; ++i) {
        const gguf_tensor_t *gt = &src.tensors[i];
        void *src_data = gguf_get_tensor_data(&src, gt);
        memcpy(data_buf + dst.tensors[i].offset, src_data, (size_t)gt->size);
    }
    dst.data = data_buf;

    rc = st_write(st_path, &dst);

    free(data_buf);
    st_free(&dst);
    gguf_free(&src);
    gguf_munmap(&gm);
    return rc;
}

/* ================================================================
 * GGUF → TQ
 *
 * Similar to safetensors → TQ: F32 tensors get ternary-quantized,
 * others are stored as raw bytes.
 * ================================================================ */

int convert_gguf_to_tq_opts(const char *gguf_path, const char *tq_path,
                            const convert_opts_t *opts) {
    gguf_mmap_t gm;
    gguf_file_t src;
    tq_header_t hdr;
    tq_tensor_t *descs = NULL;
    FILE *fp = NULL;
    uint64_t i;
    int rc = -1;
    int lz4 = convert_opts_lz4(opts);
    long desc_start, aligned;
    uint64_t data_offset = 0;
    uint32_t bits = convert_opts_bits(opts);

    if (gguf_mmap(gguf_path, &gm) != 0) return -1;
    if (gguf_parse(&gm, &src) != 0) { gguf_munmap(&gm); return -1; }

    descs = (tq_tensor_t *)calloc((size_t)src.tensor_count, sizeof(tq_tensor_t));
    if (!descs && src.tensor_count > 0) goto done;

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = TQ_MAGIC;
    hdr.version = TQ_VERSION;
    hdr.tensor_count = src.tensor_count;
    hdr.model_family_id = TQ_FAMILY_UNKNOWN;
    if (lz4) hdr.features |= TQ_FEATURE_LZ4_PER_TENSOR;

    desc_start = (long)(sizeof(tq_header_t) + (size_t)src.tensor_count * sizeof(tq_tensor_t));
    aligned = (desc_start + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;

    fp = fopen(tq_path, "wb");
    if (!fp) goto done;

    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(descs, sizeof(tq_tensor_t), (size_t)src.tensor_count, fp);
    {
        long pos = ftell(fp);
        while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }
    }

    for (i = 0; i < src.tensor_count; ++i) {
        const gguf_tensor_t *gt = &src.tensors[i];
        tq_tensor_t *td = &descs[i];
        uint64_t n_elements;
        uint8_t *buf = NULL;
        uint32_t d;

        strncpy(td->name, gt->name, sizeof(td->name) - 1);
        n_elements = 1;
        for (d = 0; d < gt->n_dims; ++d) n_elements *= gt->ne[d];
        td->rows = (uint32_t)n_elements;
        td->cols = 1;
        td->frame_offset = data_offset;

        if (gt->type == GGUF_TYPE_F32 || gt->type == GGUF_TYPE_BF16) {
            float *tmp = NULL;
            const float *fdata;

            if (gt->type == GGUF_TYPE_BF16) {
                const uint16_t *src16 =
                    (const uint16_t *)gguf_get_tensor_data(&src, gt);
                tmp = (float *)malloc(n_elements * sizeof(float));
                if (tmp) {
                    uint64_t j;
                    for (j = 0; j < n_elements; ++j)
                        tmp[j] = bf16_to_f32(src16[j]);
                }
                fdata = tmp;
            } else {
                fdata = (const float *)gguf_get_tensor_data(&src, gt);
            }

            if (fdata) {
#if defined(__ARM_NEON) && defined(TQ_WITH_NEON)
                buf = (uint8_t *)malloc(tq_packed_size(n_elements, bits));
                if (buf) quantize_f32_to_polar(fdata, buf, n_elements, td, bits);
#else
                td->b = 2;
                td->tensor_flags = 0;
                td->unpacked_size = (n_elements + 3) / 4;
                buf = (uint8_t *)calloc(1, (size_t)td->unpacked_size);
                if (buf) quantize_f32_to_ternary(fdata, buf, n_elements);
#endif
            }
            free(tmp);
        } else {
            td->b = 0;
            td->tensor_flags = TQ_TFLAG_SET_ORIG_TYPE(0, (uint32_t)gt->type);
            td->unpacked_size = gt->size;
            buf = (uint8_t *)malloc((size_t)gt->size);
            if (buf) memcpy(buf, gguf_get_tensor_data(&src, gt), (size_t)gt->size);
        }

        if (!buf) goto done;

#ifdef TQ_WITH_LZ4
        if (lz4 && td->unpacked_size > 0) {
            LZ4F_preferences_t prefs;
            size_t bound, csize;
            uint8_t *cbuf;

            memset(&prefs, 0, sizeof(prefs));
            prefs.frameInfo.contentSize = td->unpacked_size;
            bound = LZ4F_compressFrameBound(td->unpacked_size, &prefs);
            cbuf = (uint8_t *)malloc(bound);
            if (cbuf) {
                csize = LZ4F_compressFrame(cbuf, bound, buf, td->unpacked_size, &prefs);
                if (!LZ4F_isError(csize) && csize < td->unpacked_size) {
                    td->frame_size = csize;
                    fwrite(cbuf, 1, csize, fp);
                    data_offset += csize;
                    free(cbuf);
                    free(buf);
                    buf = NULL;
                } else {
                    free(cbuf);
                }
            }
        }
#endif
        if (buf) {
            td->frame_size = 0;
            fwrite(buf, 1, (size_t)td->unpacked_size, fp);
            data_offset += td->unpacked_size;
            free(buf);
        }
    }

    hdr.total_data_size = data_offset;

    if (fseek(fp, 0, SEEK_SET) != 0) goto done;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(descs, sizeof(tq_tensor_t), (size_t)src.tensor_count, fp);
    rc = 0;

done:
    if (fp) fclose(fp);
    if (rc != 0 && tq_path) remove(tq_path);
    free(descs);
    gguf_free(&src);
    gguf_munmap(&gm);
    return rc;
}

int convert_gguf_to_tq(const char *gguf_path, const char *tq_path) {
    return convert_gguf_to_tq_opts(gguf_path, tq_path, NULL);
}

/* ================================================================
 * TQ → Safetensors
 *
 * Dequantizes each tensor to F32, writes as Safetensors.
 * ================================================================ */

int convert_tq_to_safetensors(const char *tq_path, const char *st_path) {
    tq_file_t src;
    st_file_t dst;
    uint8_t *data_buf;
    uint64_t i, data_offset, total_data;
    int rc;

    if (tq_mmap(tq_path, &src) != 0) return -1;

    memset(&dst, 0, sizeof(dst));
    dst.num_tensors = (uint32_t)src.hdr->tensor_count;
    dst.tensors = (st_tensor_t *)calloc((size_t)src.hdr->tensor_count, sizeof(st_tensor_t));
    if (!dst.tensors && src.hdr->tensor_count > 0) {
        tq_munmap(&src); return -1;
    }

    /* Calculate output sizes */
    total_data = 0;
    for (i = 0; i < src.hdr->tensor_count; ++i) {
        const tq_tensor_t *tt = &src.tensors[i];
        st_tensor_t *st = &dst.tensors[i];

        st->name = (char *)malloc(strlen(tt->name) + 1);
        if (st->name) strcpy(st->name, tt->name);

        st->ndim = 2;
        st->shape[0] = tt->rows;
        st->shape[1] = tt->cols;

        if (tt->tensor_flags & TQ_TFLAG_PASSTHROUGH) {
            /* Passthrough: restore as raw bytes with closest ST dtype */
            uint32_t orig = TQ_TFLAG_GET_ORIG_TYPE(tt->tensor_flags);
            st->dtype = gguf_type_to_st((gguf_type_t)orig);
            st->size = tt->unpacked_size;
        } else {
            /* Ternary → dequantize to F32 */
            uint64_t n_elements = (uint64_t)tt->rows * tt->cols;
            st->dtype = ST_F32;
            st->size = n_elements * 4;
        }

        st->offset = total_data;
        total_data += st->size;
    }

    dst.data_size = total_data;

    data_buf = (uint8_t *)malloc((size_t)total_data);
    if (!data_buf && total_data > 0) {
        st_free(&dst); tq_munmap(&src); return -1;
    }

    /* Write tensor data into the buffer */
    for (i = 0; i < src.hdr->tensor_count; ++i) {
        const tq_tensor_t *tt = &src.tensors[i];
        data_offset = dst.tensors[i].offset;

        if (tt->tensor_flags & TQ_TFLAG_PASSTHROUGH) {
            const void *raw = tq_get_tensor_data(&src, tt);
            memcpy(data_buf + data_offset, raw, (size_t)tt->unpacked_size);
        } else {
            tq_dequant(&src, (uint32_t)i, (float *)(data_buf + data_offset));
        }
    }
    dst.data = data_buf;

    rc = st_write(st_path, &dst);

    free(data_buf);
    st_free(&dst);
    tq_munmap(&src);
    return rc;
}

/* ================================================================
 * TQ → GGUF
 *
 * Dequantizes each tensor to F32, writes as GGUF.
 * ================================================================ */

int convert_tq_to_gguf(const char *tq_path, const char *gguf_path) {
    tq_file_t src;
    gguf_file_t dst;
    uint8_t *data_buf;
    uint64_t i, total_data;
    int rc;

    if (tq_mmap(tq_path, &src) != 0) return -1;

    memset(&dst, 0, sizeof(dst));
    dst.magic = GGUF_MAGIC;
    dst.version = GGUF_VERSION;
    dst.tensor_count = src.hdr->tensor_count;
    dst.metadata_count = 0;
    dst.metadata = NULL;

    dst.tensors = (gguf_tensor_t *)calloc((size_t)src.hdr->tensor_count,
                                          sizeof(gguf_tensor_t));
    if (!dst.tensors && src.hdr->tensor_count > 0) {
        tq_munmap(&src); return -1;
    }

    total_data = 0;
    for (i = 0; i < src.hdr->tensor_count; ++i) {
        const tq_tensor_t *tt = &src.tensors[i];
        gguf_tensor_t *gt = &dst.tensors[i];

        gt->name = (char *)malloc(strlen(tt->name) + 1);
        if (gt->name) strcpy(gt->name, tt->name);

        gt->n_dims = 2;
        gt->ne[0] = tt->rows;
        gt->ne[1] = tt->cols;

        if (tt->tensor_flags & TQ_TFLAG_PASSTHROUGH) {
            /* Restore original quantized type and raw byte size */
            gt->type = (gguf_type_t)TQ_TFLAG_GET_ORIG_TYPE(tt->tensor_flags);
            gt->size = tt->unpacked_size;
        } else {
            /* PolarQuant → dequantize to BF16 (half the size of F32, no extra fidelity loss) */
            uint64_t n_elements = (uint64_t)tt->rows * tt->cols;
            gt->type = GGUF_TYPE_BF16;
            gt->size = n_elements * 2;
        }

        total_data = (total_data + 31) & ~(uint64_t)31;
        gt->offset = total_data;
        total_data += gt->size;
    }

    data_buf = (uint8_t *)calloc(1, (size_t)total_data);
    if (!data_buf && total_data > 0) {
        for (i = 0; i < src.hdr->tensor_count; ++i) free(dst.tensors[i].name);
        free(dst.tensors);
        tq_munmap(&src); return -1;
    }

    for (i = 0; i < src.hdr->tensor_count; ++i) {
        const tq_tensor_t *tt = &src.tensors[i];
        uint64_t off = dst.tensors[i].offset;

        if (tt->tensor_flags & TQ_TFLAG_PASSTHROUGH) {
            /* Copy raw bytes back */
            const void *raw = tq_get_tensor_data(&src, tt);
            memcpy(data_buf + off, raw, (size_t)tt->unpacked_size);
        } else {
            /* Dequantize to F32, then narrow to BF16 in-place */
            uint64_t n_elements = (uint64_t)tt->rows * tt->cols;
            float *tmp = (float *)malloc(n_elements * sizeof(float));
            if (tmp) {
                uint16_t *dst16 = (uint16_t *)(data_buf + off);
                uint64_t j;
                tq_dequant(&src, (uint32_t)i, tmp);
                for (j = 0; j < n_elements; ++j) {
                    uint32_t bits;
                    memcpy(&bits, &tmp[j], sizeof(bits));
                    dst16[j] = (uint16_t)(bits >> 16);
                }
                free(tmp);
            }
        }
    }
    dst.data = data_buf;

    rc = gguf_write(gguf_path, &dst);

    free(data_buf);
    for (i = 0; i < src.hdr->tensor_count; ++i) free(dst.tensors[i].name);
    free(dst.tensors);
    tq_munmap(&src);
    return rc;
}

/* ================================================================
 * Identity converter (file copy)
 * ================================================================ */

static int convert_identity(const char *input_path, const char *output_path) {
    FILE *in, *out;
    char buf[8192];
    size_t n;

    if (strcmp(input_path, output_path) == 0) return 0;

    in = fopen(input_path, "rb");
    if (!in) return -1;
    out = fopen(output_path, "wb");
    if (!out) { fclose(in); return -1; }

    while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
        fwrite(buf, 1, n, out);

    fclose(in);
    fclose(out);
    return 0;
}

/* ================================================================
 * Generic any-to-any
 * ================================================================ */

int convert_any_to_any_opts(const char *input_path, const char *output_path,
                            const convert_opts_t *opts) {
    int in_fmt = detect_format(input_path);
    if (in_fmt < 0) return -1;

    if (strstr(output_path, ".safetensors")) {
        if (in_fmt == 1) return convert_gguf_to_safetensors(input_path, output_path);
        if (in_fmt == 2) return convert_tq_to_safetensors(input_path, output_path);
        return convert_identity(input_path, output_path);
    }
    if (strstr(output_path, ".gguf")) {
        if (in_fmt == 0) return convert_safetensors_to_gguf(input_path, output_path);
        if (in_fmt == 2) return convert_tq_to_gguf(input_path, output_path);
        return convert_identity(input_path, output_path);
    }
    if (strstr(output_path, ".tq")) {
        if (in_fmt == 0) return convert_safetensors_to_tq_opts(input_path, output_path, opts);
        if (in_fmt == 1) return convert_gguf_to_tq_opts(input_path, output_path, opts);
        return convert_identity(input_path, output_path);
    }
    return -1;
}

int convert_any_to_any(const char *input_path, const char *output_path) {
    return convert_any_to_any_opts(input_path, output_path, NULL);
}

#endif /* CONVERT_IMPLEMENTATION */
