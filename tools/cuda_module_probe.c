/* Temporary LD_PRELOAD CUDA driver probe for studying kernel launch shapes.
 * Build: gcc -shared -fPIC -O2 -o cuda_module_probe.so cuda_module_probe.c -ldl
 * Records only module bytes, CUDA entry names, launch dimensions, and status.
 * Never distribute captured third-party module bytes with this project.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef int (*load_data_fn)(void **, const void *);
typedef int (*load_data_ex_fn)(void **, const void *, unsigned int, void *, void *);
typedef int (*get_function_fn)(void **, void *, const char *);
typedef int (*launch_fn)(void *, unsigned int, unsigned int, unsigned int,
                         unsigned int, unsigned int, unsigned int,
                         unsigned int, void *, void **, void **);

static void *(*next_dlsym)(void *, const char *);
static load_data_fn orig_load_data;
static load_data_ex_fn orig_load_data_ex;
static get_function_fn orig_get_function;
static launch_fn orig_launch;
static unsigned int module_number;
static unsigned int launch_count;

static size_t image_size(const void *image) {
    uint32_t magic;
    memcpy(&magic, image, sizeof(magic));
    if (magic == 0x464c457f) {
        const Elf64_Ehdr *h = image;
        if (h->e_ident[EI_CLASS] != ELFCLASS64 || h->e_shnum > 10000)
            return 0;
        size_t end = (size_t)h->e_shoff + (size_t)h->e_shentsize * h->e_shnum;
        size_t program_end = (size_t)h->e_phoff + (size_t)h->e_phentsize * h->e_phnum;
        if (program_end > end) end = program_end;
        const Elf64_Shdr *sh = (const void *)((const char *)image + h->e_shoff);
        for (unsigned int i = 0; i < h->e_shnum; ++i) {
            if (sh[i].sh_type != SHT_NOBITS) {
                size_t candidate = (size_t)sh[i].sh_offset + sh[i].sh_size;
                if (candidate > end) end = candidate;
            }
        }
        return end < 128u * 1024u * 1024u ? end : 0;
    }
    if (magic == 0x466243b1 || magic == 0xba55ed50) {
        uint32_t header_size;
        uint64_t data_size;
        memcpy(&header_size, (const char *)image + 4, sizeof(header_size));
        memcpy(&data_size, (const char *)image + 8, sizeof(data_size));
        size_t total = (size_t)header_size + (size_t)data_size;
        return total < 128u * 1024u * 1024u ? total : 0;
    }
    if (magic == 0x65762e || magic == 0x202e) {
        const char *end = memchr(image, 0, 128u * 1024u * 1024u);
        return end ? (size_t)(end - (const char *)image) + 1 : 0;
    }
    return 0;
}

static void record_image(const void *image) {
    uint32_t magic = 0;
    memcpy(&magic, image, sizeof(magic));
    size_t bytes = image_size(image);
    unsigned int n = __sync_fetch_and_add(&module_number, 1);
    fprintf(stderr, "[cuda-probe] module %u magic=%08x bytes=%zu\n", n, magic, bytes);
    if (!bytes || !getenv("CUDA_PROBE_DIR")) return;
    char path[512];
    snprintf(path, sizeof(path), "%s/module-%03u.bin", getenv("CUDA_PROBE_DIR"), n);
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return;
    const char *p = image;
    while (bytes) {
        ssize_t written = write(fd, p, bytes);
        if (written <= 0) break;
        p += written;
        bytes -= (size_t)written;
    }
    close(fd);
}

static int probe_load_data(void **module, const void *image) {
    record_image(image);
    int status = orig_load_data(module, image);
    fprintf(stderr, "[cuda-probe] cuModuleLoadData status=%d\n", status);
    return status;
}

static int probe_load_data_ex(void **module, const void *image,
                              unsigned int count, void *options, void *values) {
    record_image(image);
    int status = orig_load_data_ex(module, image, count, options, values);
    fprintf(stderr, "[cuda-probe] cuModuleLoadDataEx status=%d\n", status);
    return status;
}

static int probe_get_function(void **function, void *module, const char *name) {
    int status = orig_get_function(function, module, name);
    fprintf(stderr, "[cuda-probe] kernel %s status=%d handle=%p\n",
            name ? name : "(null)", status, status ? 0 : *function);
    return status;
}

static int probe_launch(void *function, unsigned int gx, unsigned int gy,
                        unsigned int gz, unsigned int bx, unsigned int by,
                        unsigned int bz, unsigned int shared, void *stream,
                        void **params, void **extra) {
    unsigned int n = __sync_fetch_and_add(&launch_count, 1);
    if (n < 80)
        fprintf(stderr, "[cuda-probe] launch %u handle=%p grid=%ux%ux%u "
                "block=%ux%ux%u smem=%u\n", n, function, gx, gy, gz,
                bx, by, bz, shared);
    return orig_launch(function, gx, gy, gz, bx, by, bz, shared,
                       stream, params, extra);
}

void *dlsym(void *handle, const char *symbol) {
    if (!next_dlsym)
        next_dlsym = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.34");
    void *result = next_dlsym(handle, symbol);
    if (!symbol || !result) return result;
    if (!strcmp(symbol, "cuModuleLoadData")) {
        orig_load_data = (load_data_fn)result;
        return (void *)probe_load_data;
    }
    if (!strcmp(symbol, "cuModuleLoadDataEx")) {
        orig_load_data_ex = (load_data_ex_fn)result;
        return (void *)probe_load_data_ex;
    }
    if (!strcmp(symbol, "cuModuleGetFunction")) {
        orig_get_function = (get_function_fn)result;
        return (void *)probe_get_function;
    }
    if (!strcmp(symbol, "cuLaunchKernel")) {
        orig_launch = (launch_fn)result;
        return (void *)probe_launch;
    }
    return result;
}
