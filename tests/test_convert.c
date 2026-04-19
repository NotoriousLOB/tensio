/* test_convert.c — tests for format detection and all conversion paths */

#include "safetensors.h"
#include "gguf.h"
#include "tq.h"
#include "convert.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) do { \
    tests_run++; \
    printf("  %-50s ", #name); \
    if (test_##name()) { tests_passed++; printf("PASS\n"); } \
    else { printf("FAIL\n"); } \
} while(0)

/* ----------------------------------------------------------------
 * Fixture helpers
 * ---------------------------------------------------------------- */

static const char *ST_PATH          = "/tmp/tensio_conv_test.safetensors";
static const char *ST_BF16_PATH     = "/tmp/tensio_conv_test_bf16.safetensors";
static const char *ST_3D_PATH       = "/tmp/tensio_conv_test_3d.safetensors";
static const char *ST_5D_PATH       = "/tmp/tensio_conv_test_5d.safetensors";
static const char *ST_MULTI_PATH    = "/tmp/tensio_conv_test_multi.safetensors";
static const char *GGUF_PATH        = "/tmp/tensio_conv_test.gguf";
static const char *GGUF_BF16_PATH   = "/tmp/tensio_conv_test_bf16.gguf";
static const char *TQ_PATH          = "/tmp/tensio_conv_test.tq";

static void fwrite_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void fwrite_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static int create_st_fixture(void) {
    FILE *fp;
    const char *json =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,3],"
        "\"data_offsets\":[0,24]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float data[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};

    fp = fopen(ST_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 6, fp);
    fclose(fp);
    return 0;
}

static int create_gguf_bf16_fixture(void) {
    /* GGUF file with a single BF16 tensor "w" [2,3] = 6 elements = 12 bytes */
    static const float f32s[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};
    uint16_t bf16[6];
    long pos, aligned;
    int k;
    FILE *fp;

    for (k = 0; k < 6; ++k) {
        uint32_t bits;
        memcpy(&bits, &f32s[k], sizeof(bits));
        bf16[k] = (uint16_t)(bits >> 16);
    }

    fp = fopen(GGUF_BF16_PATH, "wb");
    if (!fp) return -1;

    fwrite_u32(fp, 0x46554747u);  /* magic */
    fwrite_u32(fp, 3);             /* version */
    fwrite_u64(fp, 1);             /* tensor_count */
    fwrite_u64(fp, 0);             /* metadata_count */

    /* tensor "w": BF16 [2,3], offset=0 */
    fwrite_u64(fp, 1); fwrite("w", 1, 1, fp);
    fwrite_u32(fp, 2);             /* n_dims */
    fwrite_u64(fp, 2);             /* ne[0] */
    fwrite_u64(fp, 3);             /* ne[1] */
    fwrite_u32(fp, 30);            /* GGUF_TYPE_BF16 = 30 */
    fwrite_u64(fp, 0);             /* offset */

    pos = ftell(fp);
    aligned = (pos + 63) & ~63L;
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }

    fwrite(bf16, sizeof(uint16_t), 6, fp);
    fclose(fp);
    return 0;
}

static int create_gguf_fixture(void) {
    FILE *fp;
    float data[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};
    long pos, aligned;

    fp = fopen(GGUF_PATH, "wb");
    if (!fp) return -1;

    fwrite_u32(fp, 0x46554747u);
    fwrite_u32(fp, 3);
    fwrite_u64(fp, 1);
    fwrite_u64(fp, 0);

    /* tensor "w": F32 [2,3] */
    fwrite_u64(fp, 1); fwrite("w", 1, 1, fp);
    fwrite_u32(fp, 2);
    fwrite_u64(fp, 2);
    fwrite_u64(fp, 3);
    fwrite_u32(fp, 0); /* F32 */
    fwrite_u64(fp, 0);

    pos = ftell(fp);
    aligned = (pos + 63) & ~63L;
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }

    fwrite(data, sizeof(float), 6, fp);
    fclose(fp);
    return 0;
}

static int create_tq_fixture(void) {
    FILE *fp;
    tq_header_t hdr;
    tq_tensor_t desc;
    long pos, aligned;
    /* 2x3=6 values b=2: 6/4 = 2 bytes (round up)
     * vals: [-1, 0, +1, -1, 0, +1]
     * encoded: [0,1,2,0] = 0x24, [1,2,0,0] = 0x09 (with padding) */
    uint8_t packed[2] = {0x24, 0x09};

    memset(&hdr, 0, sizeof(hdr));
    hdr.magic = 0x46555154u;
    hdr.version = 2;
    hdr.tensor_count = 1;

    memset(&desc, 0, sizeof(desc));
    strncpy(desc.name, "w", sizeof(desc.name) - 1);
    desc.b = 2;
    desc.rows = 2;
    desc.cols = 3;
    desc.unpacked_size = 2;

    pos = (long)(sizeof(hdr) + sizeof(desc));
    aligned = (pos + 63) & ~63L;
    hdr.data_offset = (uint64_t)aligned;
    hdr.total_data_size = 2;

    fp = fopen(TQ_PATH, "wb");
    if (!fp) return -1;
    fwrite(&hdr, sizeof(hdr), 1, fp);
    fwrite(&desc, sizeof(desc), 1, fp);
    pos = ftell(fp);
    while (pos < aligned) { uint8_t z = 0; fwrite(&z, 1, 1, fp); pos++; }
    fwrite(packed, 1, 2, fp);
    fclose(fp);
    return 0;
}

/* ----------------------------------------------------------------
 * Identity conversion tests
 * ---------------------------------------------------------------- */

static int test_identity_safetensors(void) {
    const char *out = "/tmp/tensio_conv_id.safetensors";
    int ok;
    ok = (convert_any_to_any(ST_PATH, out) == 0);
    if (ok) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            ok = (f.num_tensors == 1);
            st_free(&f); st_munmap(&mm);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

static int test_identity_gguf(void) {
    const char *out = "/tmp/tensio_conv_id.gguf";
    int ok;
    ok = (convert_any_to_any(GGUF_PATH, out) == 0);
    if (ok) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            ok = (f.tensor_count == 1);
            gguf_free(&f); gguf_munmap(&mm);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

static int test_identity_tq(void) {
    const char *out = "/tmp/tensio_conv_id.tq";
    int ok;
    ok = (convert_any_to_any(TQ_PATH, out) == 0);
    if (ok) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            ok = (f.hdr->tensor_count == 1);
            tq_munmap(&f);
        } else ok = 0;
    }
    remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * Cross-format conversion tests
 * ---------------------------------------------------------------- */

static int test_st_to_gguf(void) {
    const char *out = "/tmp/tensio_conv_st2gguf.gguf";
    int ok = 0;
    if (convert_safetensors_to_gguf(ST_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            if (t && t->type == GGUF_TYPE_F32 && t->n_dims == 2 &&
                t->ne[0] == 2 && t->ne[1] == 3) {
                float *d = (float *)gguf_get_tensor_data(&f, t);
                ok = (d[0] == -1.0f && d[2] == 1.0f && d[5] == 2.0f);
            }
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_gguf_to_st(void) {
    const char *out = "/tmp/tensio_conv_gguf2st.safetensors";
    int ok = 0;
    if (convert_gguf_to_safetensors(GGUF_PATH, out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32 && t->ndim == 2) {
                float *d = (float *)st_get_tensor_data(&f, t);
                ok = (d[0] == -1.0f && d[5] == 2.0f);
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_st_to_tq(void) {
    const char *out = "/tmp/tensio_conv_st2tq.tq";
    int ok = 0;
    if (convert_safetensors_to_tq(ST_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            /* On NEON: b=4 (PolarQuant), otherwise b=2 (ternary) */
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  (f.tensors[0].b == 2 || f.tensors[0].b == 4));
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_gguf_to_tq(void) {
    const char *out = "/tmp/tensio_conv_gguf2tq.tq";
    int ok = 0;
    if (convert_gguf_to_tq(GGUF_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            /* On NEON: b=4 (PolarQuant), otherwise b=2 (ternary) */
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  (f.tensors[0].b == 2 || f.tensors[0].b == 4));
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_tq_to_st(void) {
    const char *out = "/tmp/tensio_conv_tq2st.safetensors";
    int ok = 0;
    if (convert_tq_to_safetensors(TQ_PATH, out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32 && t->ndim == 2) {
                float *d = (float *)st_get_tensor_data(&f, t);
                /* Just check values are finite (PolarQuant transformed) */
                ok = (isfinite(d[0]) && isfinite(d[1]) && isfinite(d[2]));
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_tq_to_gguf(void) {
    const char *out = "/tmp/tensio_conv_tq2gguf.gguf";
    int ok = 0;
    if (convert_tq_to_gguf(TQ_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            /* Output is now BF16 (half the size of F32, same fidelity) */
            if (t && t->type == GGUF_TYPE_BF16) {
                const uint16_t *d = (const uint16_t *)gguf_get_tensor_data(&f, t);
                /* Expand first value to F32 and check it's finite */
                uint32_t bits = (uint32_t)d[0] << 16;
                float v; memcpy(&v, &bits, sizeof(v));
                ok = isfinite(v);
            }
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * Round-trip tests: A → B → A, verify data preserved
 * ---------------------------------------------------------------- */

static int test_roundtrip_st_gguf_st(void) {
    const char *mid = "/tmp/tensio_conv_rt_mid.gguf";
    const char *out = "/tmp/tensio_conv_rt_out.safetensors";
    int ok = 0;

    if (convert_safetensors_to_gguf(ST_PATH, mid) == 0 &&
        convert_gguf_to_safetensors(mid, out) == 0) {
        st_mmap_t mm1, mm2; st_file_t f1, f2;
        if (st_mmap(ST_PATH, &mm1) == 0 && st_parse(&mm1, &f1) == 0 &&
            st_mmap(out, &mm2) == 0 && st_parse(&mm2, &f2) == 0) {
            const st_tensor_t *t1 = st_get_tensor(&f1, "w");
            const st_tensor_t *t2 = st_get_tensor(&f2, "w");
            if (t1 && t2 && t1->size == t2->size) {
                float *d1 = (float *)st_get_tensor_data(&f1, t1);
                float *d2 = (float *)st_get_tensor_data(&f2, t2);
                ok = (memcmp(d1, d2, (size_t)t1->size) == 0);
            }
            st_free(&f2); st_munmap(&mm2);
            st_free(&f1); st_munmap(&mm1);
        }
    }
    remove(mid); remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * BF16 conversion tests
 * ---------------------------------------------------------------- */

static int create_st_bf16_fixture(void) {
    /* BF16 is F32 with the low 16 bits zeroed, stored as uint16_t.
     * Values: -1.0, 0.0, 1.0, -0.5, 0.5, 2.0 */
    static const float f32s[6] = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};
    uint16_t bf16[6];
    int k;
    FILE *fp;
    const char *json =
        "{\"w\":{\"dtype\":\"BF16\",\"shape\":[2,3],"
        "\"data_offsets\":[0,12]}}";
    uint64_t json_len = (uint64_t)strlen(json);

    for (k = 0; k < 6; ++k) {
        uint32_t bits;
        memcpy(&bits, &f32s[k], sizeof(bits));
        bf16[k] = (uint16_t)(bits >> 16);
    }

    fp = fopen(ST_BF16_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(bf16, sizeof(uint16_t), 6, fp);
    fclose(fp);
    return 0;
}

static int test_st_bf16_to_tq(void) {
    const char *out = "/tmp/tensio_conv_bf16_st2tq.tq";
    int ok = 0;
    if (convert_safetensors_to_tq(ST_BF16_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  (f.tensors[0].b == 2 || f.tensors[0].b == 4));
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_st_bf16_to_tq_values(void) {
    /* Verify that BF16 → TQ → F32 round-trip yields finite values */
    const char *tq_out = "/tmp/tensio_conv_bf16_vals.tq";
    const char *st_out = "/tmp/tensio_conv_bf16_vals.safetensors";
    int ok = 0;

    if (convert_safetensors_to_tq(ST_BF16_PATH, tq_out) == 0 &&
        convert_tq_to_safetensors(tq_out, st_out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(st_out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32 && t->ndim == 2) {
                float *d = (float *)st_get_tensor_data(&f, t);
                ok = (isfinite(d[0]) && isfinite(d[1]) && isfinite(d[2]));
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(tq_out); remove(st_out);
    return ok;
}

/* ----------------------------------------------------------------
 * GGUF offset alignment tests
 *
 * GGUF spec requires tensor offsets within the data section to be
 * 32-byte aligned. Bugs only surface with multiple tensors where the
 * first tensor's size is not a multiple of 32, causing subsequent
 * offsets to drift and offset+size to exceed the file size.
 * ---------------------------------------------------------------- */

/* Two tensors: "a" is [2,3] = 24 bytes (not a multiple of 32),
 * "b" is [4,4] = 64 bytes. Without alignment, "b"'s offset would be
 * 24 instead of 32, breaking GGUF readers. */
static int create_st_multi_fixture(void) {
    const char *json =
        "{\"a\":{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,24]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[4,4],\"data_offsets\":[24,88]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float a[6]  = {-1.0f, 0.0f, 1.0f, -0.5f, 0.5f, 2.0f};
    float b[16];
    int k;
    FILE *fp;

    for (k = 0; k < 16; ++k) b[k] = (float)k * 0.1f;

    fp = fopen(ST_MULTI_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(a, sizeof(float), 6, fp);
    fwrite(b, sizeof(float), 16, fp);
    fclose(fp);
    return 0;
}

static int test_gguf_bf16_to_st_type(void) {
    /* BF16 GGUF → ST must produce ST_BF16, not ST_F32.
     * Promotion would double the reported tensor size. */
    const char *out = "/tmp/tensio_conv_bf16_gguf2st.safetensors";
    int ok = 0;
    if (convert_gguf_to_safetensors(GGUF_BF16_PATH, out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            /* Must be BF16 and size must be 6 * 2 = 12 bytes */
            if (t && t->dtype == ST_BF16 && t->size == 12)
                ok = 1;
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_st_bf16_to_gguf_type(void) {
    /* BF16 ST → GGUF must preserve BF16 type, not silently promote to F32.
     * Promotion doubles the computed tensor size, breaking offset calculations. */
    const char *out = "/tmp/tensio_conv_bf16_gguf.gguf";
    int ok = 0;
    if (convert_safetensors_to_gguf(ST_BF16_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            /* Must be BF16 (type 30) and size must be 6 * 2 = 12 bytes, not 24 */
            if (t && t->type == GGUF_TYPE_BF16 && t->size == 12 &&
                t->offset + t->size <= mm.size)
                ok = 1;
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_st_5d_to_gguf_size(void) {
    /* 5D ST → GGUF: the GGUF tensor size must match the full element count,
     * not just the first 4 dims (regression: clamping n_dims to 4 without
     * flattening caused offset+size to exceed file size for later tensors). */
    const char *out = "/tmp/tensio_conv_5d_gguf.gguf";
    int ok = 0;
    if (convert_safetensors_to_gguf(ST_5D_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *t = gguf_get_tensor(&f, "w");
            /* Full size: 4*3*2*2*2 = 96 floats = 384 bytes */
            if (t && t->size == 96 * sizeof(float) &&
                t->offset + t->size <= mm.size)
                ok = 1;
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

static int test_st_multi_to_gguf_alignment(void) {
    /* Convert 2-tensor ST (first tensor 24 bytes, not multiple of 32) to GGUF.
     * Verify second tensor's offset is 32-byte aligned and data is intact. */
    const char *out = "/tmp/tensio_conv_multi_align.gguf";
    int ok = 0;
    if (convert_safetensors_to_gguf(ST_MULTI_PATH, out) == 0) {
        gguf_mmap_t mm; gguf_file_t f;
        if (gguf_mmap(out, &mm) == 0 && gguf_parse(&mm, &f) == 0) {
            const gguf_tensor_t *tb = gguf_get_tensor(&f, "b");
            if (tb && f.tensor_count == 2 &&
                (tb->offset % 32) == 0 &&          /* must be 32-byte aligned */
                tb->offset + tb->size <= mm.size) { /* must be within file */
                float *d = (float *)gguf_get_tensor_data(&f, tb);
                ok = (d[0] == 0.0f && d[15] == 1.5f);
            }
            gguf_free(&f); gguf_munmap(&mm);
        }
    }
    remove(out);
    return ok;
}

/* ----------------------------------------------------------------
 * High-dimensional tensor tests (matches shapes seen in the wild)
 * ---------------------------------------------------------------- */

/* 3D: shape [8,1,4] = 32 elements — mirrors Qwen3 (8192,1,4) rotation tensors */
static int create_st_3d_fixture(void) {
    const char *json =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[8,1,4],"
        "\"data_offsets\":[0,128]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float data[32];
    int k;
    FILE *fp;

    for (k = 0; k < 32; ++k) data[k] = (float)(k - 16) * 0.1f;

    fp = fopen(ST_3D_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 32, fp);
    fclose(fp);
    return 0;
}

/* 5D: shape [4,3,2,2,2] = 96 elements — mirrors Qwen3 (1024,3,2,16,16) conv weights */
static int create_st_5d_fixture(void) {
    const char *json =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[4,3,2,2,2],"
        "\"data_offsets\":[0,384]}}";
    uint64_t json_len = (uint64_t)strlen(json);
    float data[96];
    int k;
    FILE *fp;

    for (k = 0; k < 96; ++k) data[k] = (float)(k - 48) * 0.02f;

    fp = fopen(ST_5D_PATH, "wb");
    if (!fp) return -1;
    fwrite(&json_len, 8, 1, fp);
    fwrite(json, 1, (size_t)json_len, fp);
    fwrite(data, sizeof(float), 96, fp);
    fclose(fp);
    return 0;
}

static int test_st_3d_to_tq(void) {
    const char *out = "/tmp/tensio_conv_3d.tq";
    int ok = 0;
    if (convert_safetensors_to_tq(ST_3D_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            /* Flattened: rows=32, cols=1 */
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  (uint64_t)f.tensors[0].rows * f.tensors[0].cols == 32);
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_st_5d_to_tq(void) {
    const char *out = "/tmp/tensio_conv_5d.tq";
    int ok = 0;
    if (convert_safetensors_to_tq(ST_5D_PATH, out) == 0) {
        tq_file_t f;
        if (tq_mmap(out, &f) == 0) {
            /* Flattened: rows=96, cols=1 */
            ok = (f.hdr->tensor_count == 1 &&
                  strcmp(f.tensors[0].name, "w") == 0 &&
                  (uint64_t)f.tensors[0].rows * f.tensors[0].cols == 96);
            tq_munmap(&f);
        }
    }
    remove(out);
    return ok;
}

static int test_st_5d_roundtrip(void) {
    /* 5D ST → TQ → ST; dequantized values must be finite */
    const char *tq_out = "/tmp/tensio_conv_5d_rt.tq";
    const char *st_out = "/tmp/tensio_conv_5d_rt.safetensors";
    int ok = 0;

    if (convert_safetensors_to_tq(ST_5D_PATH, tq_out) == 0 &&
        convert_tq_to_safetensors(tq_out, st_out) == 0) {
        st_mmap_t mm; st_file_t f;
        if (st_mmap(st_out, &mm) == 0 && st_parse(&mm, &f) == 0) {
            const st_tensor_t *t = st_get_tensor(&f, "w");
            if (t && t->dtype == ST_F32) {
                float *d = (float *)st_get_tensor_data(&f, t);
                uint64_t n = t->size / sizeof(float), j;
                ok = 1;
                for (j = 0; j < n && ok; ++j)
                    if (!isfinite(d[j])) ok = 0;
            }
            st_free(&f); st_munmap(&mm);
        }
    }
    remove(tq_out); remove(st_out);
    return ok;
}

/* ----------------------------------------------------------------
 * Error handling tests
 * ---------------------------------------------------------------- */

static int test_convert_nonexistent(void) {
    return convert_any_to_any("/tmp/no_such_file.safetensors",
                              "/tmp/out.gguf") != 0;
}

static int test_convert_unknown_ext(void) {
    return convert_any_to_any(ST_PATH, "/tmp/out.xyz") != 0;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */

int main(void) {
    printf("=== Convert Tests ===\n");
    fflush(stdout);

    if (create_st_fixture() != 0 ||
        create_st_bf16_fixture() != 0 ||
        create_st_3d_fixture() != 0 ||
        create_st_5d_fixture() != 0 ||
        create_st_multi_fixture() != 0 ||
        create_gguf_fixture() != 0 ||
        create_gguf_bf16_fixture() != 0 ||
        create_tq_fixture() != 0) {
        fprintf(stderr, "Failed to create fixtures\n");
        return 1;
    }

    /* Identity */
    TEST(identity_safetensors);
    TEST(identity_gguf);
    TEST(identity_tq);

    /* Cross-format */
    TEST(st_to_gguf);
    TEST(gguf_to_st);
    TEST(st_to_tq);
    TEST(gguf_to_tq);
    TEST(tq_to_st);
    TEST(tq_to_gguf);

    /* Round-trip */
    TEST(roundtrip_st_gguf_st);

    /* BF16 */
    TEST(st_bf16_to_tq);
    TEST(st_bf16_to_tq_values);

    /* GGUF type mapping, offset alignment, and 5D size */
    TEST(st_bf16_to_gguf_type);
    TEST(gguf_bf16_to_st_type);
    TEST(st_5d_to_gguf_size);
    TEST(st_multi_to_gguf_alignment);

    /* High-dimensional tensors */
    TEST(st_3d_to_tq);
    TEST(st_5d_to_tq);
    TEST(st_5d_roundtrip);

    /* Error handling */
    TEST(convert_nonexistent);
    TEST(convert_unknown_ext);

    remove(ST_PATH);
    remove(ST_BF16_PATH);
    remove(ST_3D_PATH);
    remove(ST_5D_PATH);
    remove(ST_MULTI_PATH);
    remove(GGUF_PATH);
    remove(GGUF_BF16_PATH);
    remove(TQ_PATH);

    printf("\n%d/%d tests passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
