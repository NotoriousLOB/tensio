/* tensio CLI — minimal entry point
 * Full CLI implemented in Phase 7
 */

/* Implementation macros are set via CMake target_compile_definitions.
 * For standalone compilation, define them on the command line:
 *   -DSAFETENSORS_IMPLEMENTATION -DGGUF_IMPLEMENTATION ...
 */

#include "tensio.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <command> [args...]\n"
        "\n"
        "Commands:\n"
        "  info    <file>          Show file format and tensor summary\n"
        "  list    <file>          List all tensors\n"
        "  convert <in> <out> [--bits N]  Convert between formats (N=2-8, default 4, TQ only)\n"
        "  copy-meta <src.gguf> <dst.gguf>  Copy KV metadata from src into dst (in-place)\n"
        "\n"
        "Supported formats: .safetensors, .gguf, .tq\n",
        prog);
}

static int cmd_info(const char *path) {
    /* Try GGUF first (has magic) */
    {
        gguf_mmap_t mm;
        if (gguf_mmap(path, &mm) == 0) {
            gguf_file_t f;
            if (gguf_parse(&mm, &f) == 0) {
                printf("Format:    GGUF v%u\n", f.version);
                printf("Tensors:   %llu\n", (unsigned long long)f.tensor_count);
                printf("Metadata:  %llu\n", (unsigned long long)f.metadata_count);
                gguf_free(&f);
                gguf_munmap(&mm);
                return 0;
            }
            gguf_munmap(&mm);
        }
    }

    /* Try TQ */
    {
        tq_file_t f;
        if (tq_mmap(path, &f) == 0) {
            printf("Format:    TQ v%u\n", f.hdr->version);
            printf("Tensors:   %llu\n", (unsigned long long)f.hdr->tensor_count);
            printf("Family:    %u\n", f.hdr->model_family_id);
            printf("Features:  0x%llx\n", (unsigned long long)f.hdr->features);
            tq_munmap(&f);
            return 0;
        }
    }

    /* Try Safetensors (no magic, assume if others fail) */
    {
        st_mmap_t mm;
        if (st_mmap(path, &mm) == 0) {
            st_file_t f;
            if (st_parse(&mm, &f) == 0) {
                printf("Format:    Safetensors\n");
                printf("Tensors:   %u\n", f.num_tensors);
                printf("Header:    %llu bytes\n",
                       (unsigned long long)f.header_len);
                printf("Data:      %llu bytes\n",
                       (unsigned long long)f.data_size);
                st_free(&f);
                st_munmap(&mm);
                return 0;
            }
            st_munmap(&mm);
        }
    }

    fprintf(stderr, "Error: could not open or parse '%s'\n", path);
    return 1;
}

static int cmd_list(const char *path) {
    /* Try GGUF */
    {
        gguf_mmap_t mm;
        if (gguf_mmap(path, &mm) == 0) {
            gguf_file_t f;
            if (gguf_parse(&mm, &f) == 0) {
                uint64_t i;
                printf("%-60s  %-8s  %s\n", "NAME", "TYPE", "SIZE");
                for (i = 0; i < f.tensor_count; ++i) {
                    printf("%-60s  %-8u  %llu\n",
                           f.tensors[i].name,
                           (unsigned)f.tensors[i].type,
                           (unsigned long long)f.tensors[i].size);
                }
                gguf_free(&f);
                gguf_munmap(&mm);
                return 0;
            }
            gguf_munmap(&mm);
        }
    }

    /* Try TQ */
    {
        tq_file_t f;
        if (tq_mmap(path, &f) == 0) {
            uint64_t i;
            printf("%-60s  %s  %s  %s\n", "NAME", "B", "ROWS", "COLS");
            for (i = 0; i < f.hdr->tensor_count; ++i) {
                printf("%-60s  %u  %u  %u\n",
                       f.tensors[i].name,
                       f.tensors[i].b,
                       f.tensors[i].rows,
                       f.tensors[i].cols);
            }
            tq_munmap(&f);
            return 0;
        }
    }

    /* Try Safetensors */
    {
        st_mmap_t mm;
        if (st_mmap(path, &mm) == 0) {
            st_file_t f;
            if (st_parse(&mm, &f) == 0) {
                uint32_t i;
                printf("%-60s  %-6s  %s\n", "NAME", "DTYPE", "SIZE");
                for (i = 0; i < f.num_tensors; ++i) {
                    printf("%-60s  %-6u  %llu\n",
                           f.tensors[i].name,
                           (unsigned)f.tensors[i].dtype,
                           (unsigned long long)f.tensors[i].size);
                }
                st_free(&f);
                st_munmap(&mm);
                return 0;
            }
            st_munmap(&mm);
        }
    }

    fprintf(stderr, "Error: could not open or parse '%s'\n", path);
    return 1;
}

static int cmd_convert(const char *in, const char *out, uint32_t bits) {
    int rc;
    convert_opts_t opts;
    memset(&opts, 0, sizeof(opts));
    opts.bits_per_weight = bits; /* 0 = library default (4) */

    /* Enable LZ4 per-tensor compression by default when writing TQ */
#ifdef TQ_WITH_LZ4
    if (strstr(out, ".tq"))
        opts.use_lz4 = 1;
#endif

    rc = convert_any_to_any_opts(in, out, &opts);
    if (rc != 0) {
        fprintf(stderr, "Error: conversion failed\n");
        return 1;
    }
    printf("Converted %s -> %s", in, out);
    if (strstr(out, ".tq")) {
        uint32_t actual = bits ? bits : 4;
        printf(" (%u-bit", actual);
#ifdef TQ_WITH_LZ4
        if (opts.use_lz4) printf(", LZ4 compressed");
#endif
        printf(")");
    }
    printf("\n");
    return 0;
}

/* Copy KV metadata from src GGUF into dst GGUF in-place.
 *
 * Strategy: the GGUF layout is:
 *   [header 24 bytes][KV section][tensor infos][padding][data]
 *
 * We splice src's raw KV bytes into dst by rebuilding the file:
 *   new header (src kv_count) + src KV bytes + dst tensor infos + dst data
 */
static int cmd_copy_meta(const char *src_path, const char *dst_path) {
    gguf_mmap_t sm, dm;
    gguf_file_t sf, df;
    FILE *fp = NULL;
    uint8_t *tmp_path_buf = NULL;
    int rc = 1;

    if (gguf_mmap(src_path, &sm) != 0) {
        fprintf(stderr, "Error: cannot open src '%s'\n", src_path); return 1;
    }
    if (gguf_parse(&sm, &sf) != 0) {
        fprintf(stderr, "Error: cannot parse src '%s'\n", src_path);
        gguf_munmap(&sm); return 1;
    }
    if (gguf_mmap(dst_path, &dm) != 0) {
        fprintf(stderr, "Error: cannot open dst '%s'\n", dst_path);
        gguf_free(&sf); gguf_munmap(&sm); return 1;
    }
    if (gguf_parse(&dm, &df) != 0) {
        fprintf(stderr, "Error: cannot parse dst '%s'\n", dst_path);
        gguf_munmap(&dm); gguf_free(&sf); gguf_munmap(&sm); return 1;
    }

    /* Use kv_end pointers recorded by gguf_parse (avoids fragile manual re-walk) */
    {
        /* src KV bytes: between header (24 bytes) and kv_end */
        const uint8_t *src_kv_start = sm.base + 24;
        size_t src_kv_len = (size_t)(sf.kv_end - src_kv_start);

        /* dst tensor info bytes: from kv_end to data (includes alignment padding) */
        const uint8_t *dst_ti_start = df.kv_end;
        size_t dst_ti_len = (size_t)(df.data - dst_ti_start);

        /* Write to a temp file then rename */
        size_t tmp_len = strlen(dst_path) + 5;
        tmp_path_buf = (uint8_t *)malloc(tmp_len);
        if (!tmp_path_buf) goto done;
        snprintf((char *)tmp_path_buf, tmp_len, "%s.tmp", dst_path);

        fp = fopen((char *)tmp_path_buf, "wb");
        if (!fp) goto done;

        /* New header: same magic/version/tensor_count as dst, but src's kv_count */
        {
            uint32_t magic, version;
            uint64_t tensor_count;
            memcpy(&magic,        dm.base,     4);
            memcpy(&version,      dm.base + 4, 4);
            memcpy(&tensor_count, dm.base + 8, 8);
            fwrite(&magic,              4, 1, fp);
            fwrite(&version,            4, 1, fp);
            fwrite(&tensor_count,       8, 1, fp);
            fwrite(&sf.metadata_count,  8, 1, fp);
        }

        /* src KV section */
        fwrite(src_kv_start, 1, src_kv_len, fp);

        /* dst tensor infos + alignment padding */
        fwrite(dst_ti_start, 1, dst_ti_len, fp);

        /* dst data section: total size = max(offset + size) across all tensors */
        {
            uint64_t total = 0, ti;
            for (ti = 0; ti < df.tensor_count; ++ti) {
                uint64_t end = df.tensors[ti].offset + df.tensors[ti].size;
                if (end > total) total = end;
            }
            fwrite(df.data, 1, (size_t)total, fp);
        }

        fclose(fp); fp = NULL;

        if (rename((char *)tmp_path_buf, dst_path) != 0) {
            fprintf(stderr, "Error: rename failed\n");
            remove((char *)tmp_path_buf);
            goto done;
        }

        printf("Copied %llu KV entries from '%s' into '%s'\n",
               (unsigned long long)sf.metadata_count, src_path, dst_path);
        rc = 0;
    }

done:
    if (fp) { fclose(fp); remove((char *)tmp_path_buf); }
    free(tmp_path_buf);
    gguf_free(&df); gguf_munmap(&dm);
    gguf_free(&sf); gguf_munmap(&sm);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "info") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return cmd_info(argv[2]);
    }
    if (strcmp(argv[1], "list") == 0) {
        if (argc < 3) { print_usage(argv[0]); return 1; }
        return cmd_list(argv[2]);
    }
    if (strcmp(argv[1], "convert") == 0) {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        uint32_t bits = 0;
        int i;
        for (i = 4; i + 1 < argc; i++) {
            if (strcmp(argv[i], "--bits") == 0) {
                int v = atoi(argv[i + 1]);
                if (v >= 2 && v <= 8) { bits = (uint32_t)v; i++; }
                else {
                    fprintf(stderr, "Error: --bits must be between 2 and 8\n");
                    return 1;
                }
            }
        }
        return cmd_convert(argv[2], argv[3], bits);
    }
    if (strcmp(argv[1], "copy-meta") == 0) {
        if (argc < 4) { print_usage(argv[0]); return 1; }
        return cmd_copy_meta(argv[2], argv[3]);
    }

    fprintf(stderr, "Unknown command: %s\n", argv[1]);
    print_usage(argv[0]);
    return 1;
}
