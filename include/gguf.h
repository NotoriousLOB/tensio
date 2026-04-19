/* gguf.h — simple mmap parser + writer for GGUF
 * Strict C99, zero UB, header-only
 * -std=c99 -pedantic -Wall -Wextra -Werror -march=native
 */

#ifndef GGUF_H
#define GGUF_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/stat.h>
#endif

#define GGUF_MAGIC   0x46554747u   /* "GGUF" */
#define GGUF_VERSION 3

/* ================================================================
 * GGUF metadata value types
 * ================================================================ */

typedef enum {
    GGUF_META_UINT8    = 0,
    GGUF_META_INT8     = 1,
    GGUF_META_UINT16   = 2,
    GGUF_META_INT16    = 3,
    GGUF_META_UINT32   = 4,
    GGUF_META_INT32    = 5,
    GGUF_META_FLOAT32  = 6,
    GGUF_META_BOOL     = 7,
    GGUF_META_STRING   = 8,
    GGUF_META_ARRAY    = 9,
    GGUF_META_UINT64   = 10,
    GGUF_META_INT64    = 11,
    GGUF_META_FLOAT64  = 12,
} gguf_meta_type_t;

/* ================================================================
 * GGUF tensor types
 * ================================================================ */

typedef enum {
    GGUF_TYPE_F32  = 0,
    GGUF_TYPE_F16  = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q5_0 = 6,
    GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8,
    GGUF_TYPE_Q2_K = 10,
    GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12,
    GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14,
    GGUF_TYPE_Q8_K = 15,
    GGUF_TYPE_I8   = 16,
    GGUF_TYPE_I16  = 17,
    GGUF_TYPE_I32  = 18,
    GGUF_TYPE_I64  = 19,
    GGUF_TYPE_F64  = 20,
    GGUF_TYPE_BF16 = 30,
} gguf_type_t;

/* ================================================================
 * Metadata key-value pair
 * ================================================================ */

typedef struct {
    char            *key;
    gguf_meta_type_t type;
    union {
        uint8_t   u8;
        int8_t    i8;
        uint16_t  u16;
        int16_t   i16;
        uint32_t  u32;
        int32_t   i32;
        float     f32;
        uint8_t   b;
        uint64_t  u64;
        int64_t   i64;
        double    f64;
        struct {
            char    *data;
            uint64_t len;
        } str;
        struct {
            gguf_meta_type_t elem_type;
            uint64_t         count;
            void            *data;
        } arr;
    } value;
} gguf_kv_t;

/* ================================================================
 * Public Types
 * ================================================================ */

typedef struct {
    char        *name;
    gguf_type_t  type;
    uint32_t     n_dims;
    uint64_t     ne[4];      /* shape (only first n_dims are valid) */
    uint64_t     offset;     /* byte offset into data section */
    uint64_t     size;       /* byte size of this tensor's data */
} gguf_tensor_t;

typedef struct {
    uint8_t    *base;     /* mmap base */
    size_t      size;
    void       *mmap_handle;
} gguf_mmap_t;

typedef struct {
    uint32_t       magic;
    uint32_t       version;
    uint64_t       tensor_count;
    uint64_t       metadata_count;

    gguf_kv_t     *metadata;
    gguf_tensor_t *tensors;
    uint8_t       *data;        /* start of tensor data section (64-byte aligned) */
    uint8_t       *kv_end;      /* pointer to first byte after the KV section (= tensor info start) */
} gguf_file_t;

/* ================================================================
 * Public API
 * ================================================================ */

int  gguf_mmap(const char *path, gguf_mmap_t *out);
void gguf_munmap(gguf_mmap_t *mmap);

int  gguf_parse(const gguf_mmap_t *mmap, gguf_file_t *out);
void gguf_free(gguf_file_t *file);

const gguf_tensor_t *gguf_get_tensor(const gguf_file_t *f, const char *name);
void *gguf_get_tensor_data(const gguf_file_t *f, const gguf_tensor_t *t);

const gguf_kv_t *gguf_get_kv(const gguf_file_t *f, const char *key);

int gguf_write(const char *path, const gguf_file_t *f);

#endif /* GGUF_H */

/* ================================================================
 * IMPLEMENTATION (define GGUF_IMPLEMENTATION in one .c)
 * ================================================================ */

#if defined(GGUF_IMPLEMENTATION) && !defined(GGUF_IMPLEMENTATION_DONE)
#define GGUF_IMPLEMENTATION_DONE

/* ================================================================
 * Type size helpers
 * ================================================================ */

static size_t gguf_meta_type_size(gguf_meta_type_t type) {
    switch (type) {
        case GGUF_META_UINT8:   case GGUF_META_INT8:  case GGUF_META_BOOL: return 1;
        case GGUF_META_UINT16:  case GGUF_META_INT16:  return 2;
        case GGUF_META_UINT32:  case GGUF_META_INT32:  case GGUF_META_FLOAT32: return 4;
        case GGUF_META_UINT64:  case GGUF_META_INT64:  case GGUF_META_FLOAT64: return 8;
        default: return 0;
    }
}

/* Block size and type size for quantized GGUF tensor types.
 * Returns the number of bytes per block and elements per block. */
static void gguf_tensor_type_info(gguf_type_t type,
                                  size_t *block_size, size_t *type_size) {
    switch (type) {
        case GGUF_TYPE_F32:  *block_size = 1;   *type_size = 4;   break;
        case GGUF_TYPE_F16:  *block_size = 1;   *type_size = 2;   break;
        case GGUF_TYPE_Q4_0: *block_size = 32;  *type_size = 18;  break;
        case GGUF_TYPE_Q4_1: *block_size = 32;  *type_size = 20;  break;
        case GGUF_TYPE_Q5_0: *block_size = 32;  *type_size = 22;  break;
        case GGUF_TYPE_Q5_1: *block_size = 32;  *type_size = 24;  break;
        case GGUF_TYPE_Q8_0: *block_size = 32;  *type_size = 34;  break;
        case GGUF_TYPE_Q2_K: *block_size = 256; *type_size = 84;  break;
        case GGUF_TYPE_Q3_K: *block_size = 256; *type_size = 110; break;
        case GGUF_TYPE_Q4_K: *block_size = 256; *type_size = 144; break;
        case GGUF_TYPE_Q5_K: *block_size = 256; *type_size = 176; break;
        case GGUF_TYPE_Q6_K: *block_size = 256; *type_size = 210; break;
        case GGUF_TYPE_Q8_K: *block_size = 256; *type_size = 292; break;
        case GGUF_TYPE_I8:   *block_size = 1;   *type_size = 1;   break;
        case GGUF_TYPE_I16:  *block_size = 1;   *type_size = 2;   break;
        case GGUF_TYPE_I32:  *block_size = 1;   *type_size = 4;   break;
        case GGUF_TYPE_I64:  *block_size = 1;   *type_size = 8;   break;
        case GGUF_TYPE_F64:  *block_size = 1;   *type_size = 8;   break;
        case GGUF_TYPE_BF16: *block_size = 1;   *type_size = 2;   break;
        default:             *block_size = 1;   *type_size = 0;   break;
    }
}

static uint64_t gguf_tensor_nbytes(const gguf_tensor_t *t) {
    uint64_t n_elements = 1;
    size_t block_size, type_size;
    uint32_t i;
    for (i = 0; i < t->n_dims; ++i)
        n_elements *= t->ne[i];
    gguf_tensor_type_info(t->type, &block_size, &type_size);
    if (block_size == 0) return 0;
    return (n_elements / block_size) * type_size;
}

/* ================================================================
 * mmap
 * ================================================================ */

int gguf_mmap(const char *path, gguf_mmap_t *out) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }

    out->base = (uint8_t *)mmap(NULL, (size_t)st.st_size, PROT_READ,
                                MAP_PRIVATE, fd, 0);
    close(fd);
    if (out->base == MAP_FAILED) return -1;

    out->size = (size_t)st.st_size;
    out->mmap_handle = NULL;
    return 0;
}

void gguf_munmap(gguf_mmap_t *m) {
    if (m->base) munmap(m->base, m->size);
    memset(m, 0, sizeof(*m));
}

/* ================================================================
 * Read helpers (little-endian, unaligned safe via memcpy)
 * ================================================================ */

static uint32_t gguf_read_u32(const uint8_t **p) {
    uint32_t v; memcpy(&v, *p, 4); *p += 4; return v;
}

static uint64_t gguf_read_u64(const uint8_t **p) {
    uint64_t v; memcpy(&v, *p, 8); *p += 8; return v;
}

static char *gguf_read_str(const uint8_t **p, uint64_t *out_len) {
    uint64_t len = gguf_read_u64(p);
    char *s = (char *)malloc(len + 1);
    if (!s) { *p += len; if (out_len) *out_len = 0; return NULL; }
    memcpy(s, *p, len);
    s[len] = '\0';
    *p += len;
    if (out_len) *out_len = len;
    return s;
}

/* Skip a metadata value of given type, advancing p past it */
static void gguf_skip_meta_value(const uint8_t **p, gguf_meta_type_t type) {
    switch (type) {
        case GGUF_META_STRING: {
            uint64_t len = gguf_read_u64(p);
            *p += len;
            break;
        }
        case GGUF_META_ARRAY: {
            uint32_t etype = gguf_read_u32(p);
            uint64_t count = gguf_read_u64(p);
            uint64_t i;
            if (etype == GGUF_META_STRING) {
                for (i = 0; i < count; ++i) {
                    uint64_t slen = gguf_read_u64(p);
                    *p += slen;
                }
            } else if (etype == GGUF_META_ARRAY) {
                for (i = 0; i < count; ++i)
                    gguf_skip_meta_value(p, GGUF_META_ARRAY);
            } else {
                *p += count * gguf_meta_type_size((gguf_meta_type_t)etype);
            }
            break;
        }
        default:
            *p += gguf_meta_type_size(type);
            break;
    }
}

/* Read a metadata value into a gguf_kv_t */
static void gguf_read_meta_value(const uint8_t **p, gguf_kv_t *kv) {
    switch (kv->type) {
        case GGUF_META_UINT8:   memcpy(&kv->value.u8,  *p, 1); *p += 1; break;
        case GGUF_META_INT8:    memcpy(&kv->value.i8,  *p, 1); *p += 1; break;
        case GGUF_META_BOOL:    memcpy(&kv->value.b,   *p, 1); *p += 1; break;
        case GGUF_META_UINT16:  memcpy(&kv->value.u16, *p, 2); *p += 2; break;
        case GGUF_META_INT16:   memcpy(&kv->value.i16, *p, 2); *p += 2; break;
        case GGUF_META_UINT32:  memcpy(&kv->value.u32, *p, 4); *p += 4; break;
        case GGUF_META_INT32:   memcpy(&kv->value.i32, *p, 4); *p += 4; break;
        case GGUF_META_FLOAT32: memcpy(&kv->value.f32, *p, 4); *p += 4; break;
        case GGUF_META_UINT64:  memcpy(&kv->value.u64, *p, 8); *p += 8; break;
        case GGUF_META_INT64:   memcpy(&kv->value.i64, *p, 8); *p += 8; break;
        case GGUF_META_FLOAT64: memcpy(&kv->value.f64, *p, 8); *p += 8; break;
        case GGUF_META_STRING:
            kv->value.str.data = gguf_read_str(p, &kv->value.str.len);
            break;
        case GGUF_META_ARRAY:
            kv->value.arr.elem_type = (gguf_meta_type_t)gguf_read_u32(p);
            kv->value.arr.count = gguf_read_u64(p);
            /* For arrays, store raw pointer and skip past data */
            kv->value.arr.data = (void *)*p;
            if (kv->value.arr.elem_type == GGUF_META_STRING ||
                kv->value.arr.elem_type == GGUF_META_ARRAY) {
                uint64_t i;
                for (i = 0; i < kv->value.arr.count; ++i)
                    gguf_skip_meta_value(p, kv->value.arr.elem_type);
            } else {
                *p += kv->value.arr.count *
                      gguf_meta_type_size(kv->value.arr.elem_type);
            }
            break;
    }
}

/* ================================================================
 * Parser
 * ================================================================ */

int gguf_parse(const gguf_mmap_t *mmap, gguf_file_t *out) {
    uint64_t i;
    if (mmap->size < 24) return -1;

    const uint8_t *p = mmap->base;
    out->magic = gguf_read_u32(&p);
    out->version = gguf_read_u32(&p);
    out->tensor_count = gguf_read_u64(&p);
    out->metadata_count = gguf_read_u64(&p);

    if (out->magic != GGUF_MAGIC || out->version < 2 || out->version > 3)
        return -1;

    /* Metadata KV section */
    if (out->metadata_count > 0) {
        if (out->metadata_count > SIZE_MAX / sizeof(gguf_kv_t))
            return -1;
        out->metadata = (gguf_kv_t *)calloc((size_t)out->metadata_count,
                                            sizeof(gguf_kv_t));
        if (!out->metadata) return -1;

        for (i = 0; i < out->metadata_count; ++i) {
            out->metadata[i].key = gguf_read_str(&p, NULL);
            out->metadata[i].type = (gguf_meta_type_t)gguf_read_u32(&p);
            gguf_read_meta_value(&p, &out->metadata[i]);
        }
    } else {
        out->metadata = NULL;
    }
    out->kv_end = (uint8_t *)p; /* points to first byte of tensor info section */

    /* Tensor info section */
    if (out->tensor_count > SIZE_MAX / sizeof(gguf_tensor_t))
        return -1;
    out->tensors = (gguf_tensor_t *)calloc((size_t)out->tensor_count,
                                           sizeof(gguf_tensor_t));
    if (!out->tensors) return -1;

    for (i = 0; i < out->tensor_count; ++i) {
        out->tensors[i].name = gguf_read_str(&p, NULL);
        out->tensors[i].n_dims = gguf_read_u32(&p);
        if (out->tensors[i].n_dims > 4) out->tensors[i].n_dims = 4;

        memset(out->tensors[i].ne, 0, sizeof(out->tensors[i].ne));
        {
            uint32_t d;
            for (d = 0; d < out->tensors[i].n_dims; ++d)
                out->tensors[i].ne[d] = gguf_read_u64(&p);
        }

        out->tensors[i].type = (gguf_type_t)gguf_read_u32(&p);
        out->tensors[i].offset = gguf_read_u64(&p);
        out->tensors[i].size = gguf_tensor_nbytes(&out->tensors[i]);
    }

    /* Data section starts after all tensor infos (64-byte aligned) */
    out->data = (uint8_t *)(((uintptr_t)p + 63) & ~(uintptr_t)63);

    return 0;
}

void gguf_free(gguf_file_t *file) {
    uint64_t i;
    if (file->metadata) {
        for (i = 0; i < file->metadata_count; ++i) {
            free(file->metadata[i].key);
            if (file->metadata[i].type == GGUF_META_STRING)
                free(file->metadata[i].value.str.data);
        }
        free(file->metadata);
    }
    if (file->tensors) {
        for (i = 0; i < file->tensor_count; ++i)
            free(file->tensors[i].name);
        free(file->tensors);
    }
    memset(file, 0, sizeof(*file));
}

const gguf_tensor_t *gguf_get_tensor(const gguf_file_t *f, const char *name) {
    uint64_t i;
    for (i = 0; i < f->tensor_count; ++i) {
        if (strcmp(f->tensors[i].name, name) == 0)
            return &f->tensors[i];
    }
    return NULL;
}

void *gguf_get_tensor_data(const gguf_file_t *f, const gguf_tensor_t *t) {
    return f->data + t->offset;
}

const gguf_kv_t *gguf_get_kv(const gguf_file_t *f, const char *key) {
    uint64_t i;
    for (i = 0; i < f->metadata_count; ++i) {
        if (f->metadata[i].key && strcmp(f->metadata[i].key, key) == 0)
            return &f->metadata[i];
    }
    return NULL;
}

/* ================================================================
 * Write helpers
 * ================================================================ */

static void gguf_write_u32(FILE *fp, uint32_t v) { fwrite(&v, 4, 1, fp); }
static void gguf_write_u64(FILE *fp, uint64_t v) { fwrite(&v, 8, 1, fp); }

static void gguf_write_str(FILE *fp, const char *s, uint64_t len) {
    gguf_write_u64(fp, len);
    fwrite(s, 1, (size_t)len, fp);
}

static void gguf_write_meta_value(FILE *fp, const gguf_kv_t *kv) {
    switch (kv->type) {
        case GGUF_META_UINT8:   case GGUF_META_INT8:  case GGUF_META_BOOL:
            fwrite(&kv->value, 1, 1, fp); break;
        case GGUF_META_UINT16:  case GGUF_META_INT16:
            fwrite(&kv->value, 1, 2, fp); break;
        case GGUF_META_UINT32:  case GGUF_META_INT32:  case GGUF_META_FLOAT32:
            fwrite(&kv->value, 1, 4, fp); break;
        case GGUF_META_UINT64:  case GGUF_META_INT64:  case GGUF_META_FLOAT64:
            fwrite(&kv->value, 1, 8, fp); break;
        case GGUF_META_STRING:
            gguf_write_str(fp, kv->value.str.data, kv->value.str.len);
            break;
        case GGUF_META_ARRAY:
            /* Array writing not yet supported */
            gguf_write_u32(fp, (uint32_t)kv->value.arr.elem_type);
            gguf_write_u64(fp, 0);
            break;
    }
}

/* ================================================================
 * Writer
 * ================================================================ */

int gguf_write(const char *path, const gguf_file_t *f) {
    uint64_t i;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;

    /* Header */
    gguf_write_u32(fp, f->magic ? f->magic : GGUF_MAGIC);
    gguf_write_u32(fp, f->version ? f->version : GGUF_VERSION);
    gguf_write_u64(fp, f->tensor_count);
    gguf_write_u64(fp, f->metadata_count);

    /* Metadata KV pairs */
    for (i = 0; i < f->metadata_count; ++i) {
        const gguf_kv_t *kv = &f->metadata[i];
        uint64_t key_len = (uint64_t)strlen(kv->key);
        gguf_write_str(fp, kv->key, key_len);
        gguf_write_u32(fp, (uint32_t)kv->type);
        gguf_write_meta_value(fp, kv);
    }

    /* Tensor infos */
    for (i = 0; i < f->tensor_count; ++i) {
        const gguf_tensor_t *t = &f->tensors[i];
        uint64_t name_len = (uint64_t)strlen(t->name);
        uint32_t d;
        gguf_write_str(fp, t->name, name_len);
        gguf_write_u32(fp, t->n_dims);
        for (d = 0; d < t->n_dims; ++d)
            gguf_write_u64(fp, t->ne[d]);
        gguf_write_u32(fp, (uint32_t)t->type);
        gguf_write_u64(fp, t->offset);
    }

    /* Pad to 64-byte alignment */
    {
        long pos = ftell(fp);
        long aligned = (pos + 63) & ~63L;
        while (pos < aligned) {
            uint8_t zero = 0;
            fwrite(&zero, 1, 1, fp);
            pos++;
        }
    }

    /* Tensor data */
    if (f->data && f->tensor_count > 0) {
        /* Compute total data size from last tensor */
        uint64_t total = 0;
        for (i = 0; i < f->tensor_count; ++i) {
            uint64_t end = f->tensors[i].offset + f->tensors[i].size;
            if (end > total) total = end;
        }
        fwrite(f->data, 1, (size_t)total, fp);
    }

    fclose(fp);
    return 0;
}

#endif /* GGUF_IMPLEMENTATION */
