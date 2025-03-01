/* common.h - common code that can be reused between device driver
              implementations

   Copyright (c) 2012-2019 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#ifndef POCL_COMMON_H
#define POCL_COMMON_H

#include "pocl.h"
#include "utlist.h"

#define __CBUILD__
#include "pocl_image_types.h"
#undef __CBUILD__

#define XSETUP_DEVICE_CL_VERSION(D, A, B)                                     \
  D->version_as_int = (A * 100) + (B * 10);                                   \
  D->version_as_cl = CL_MAKE_VERSION (A, B, 0);                               \
  D->version = "OpenCL " #A "." #B " PoCL";                                   \
  D->opencl_c_version_as_opt = "CL" #A "." #B;                                \
  D->opencl_c_version_as_cl = CL_MAKE_VERSION (A, B, 0);

#define SETUP_DEVICE_CL_VERSION(D, a, b) XSETUP_DEVICE_CL_VERSION (D, a, b)

#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_CHAR    1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT   1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_INT     1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_LONG    1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_FLOAT   1
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_DOUBLE  1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_CHAR       1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT      1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_INT        1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_LONG       1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_FLOAT      1
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_DOUBLE     1

/* Half is internally represented as short */
#define POCL_DEVICES_PREFERRED_VECTOR_WIDTH_HALF POCL_DEVICES_PREFERRED_VECTOR_WIDTH_SHORT
#define POCL_DEVICES_NATIVE_VECTOR_WIDTH_HALF POCL_DEVICES_NATIVE_VECTOR_WIDTH_SHORT

#ifdef __cplusplus
extern "C" {
#endif

  POCL_EXPORT
  const char *cmd_type_str(cl_command_type cmd_type);

  typedef struct pocl_dlhandle_cache_item pocl_dlhandle_cache_item;
  struct pocl_dlhandle_cache_item
  {
    pocl_kernel_hash_t hash;

    /* The specialization properties. */
    /* The local dimensions. */
    size_t local_wgs[3];
    /* If global offset must be zero for this WG function version. */
    int goffs_zero;
    int specialize;
    /* Maximum grid dimension this WG function works with. */
    size_t max_grid_dim_width;

    void *wg;
    void *dlhandle;
    pocl_dlhandle_cache_item *next;
    pocl_dlhandle_cache_item *prev;
    unsigned ref_count;
  };

POCL_EXPORT
char *pocl_cpu_build_hash (cl_device_id device);

POCL_EXPORT
void pocl_broadcast (cl_event event);

  // pocl_lock_t pocl_dlhandle_lock;

  POCL_EXPORT
  void pocl_fill_dev_image_t (dev_image_t *di, struct pocl_argument *parg,
                              cl_device_id device);

  POCL_EXPORT
  void pocl_fill_dev_sampler_t (dev_sampler_t *ds, struct pocl_argument *parg);

  POCL_EXPORT
  void pocl_exec_command (_cl_command_node *node);

  POCL_EXPORT
  void pocl_init_dlhandle_cache ();

  POCL_EXPORT
  char *pocl_check_kernel_disk_cache (_cl_command_node *cmd, int specialized);

  POCL_EXPORT
  size_t pocl_max_grid_dim_width (const ulong local_size[3],
                                  const ulong num_groups[3]);

  POCL_EXPORT
  size_t pocl_cmd_max_grid_dim_width (_cl_command_run *cmd);

  POCL_EXPORT
  void *pocl_check_kernel_dlhandle_cache (_cl_command_node *command, int retain,
                                         int specialize, char *dylib_name);

  /* Look for a dlhandle in the dlhandle cache for the given kernel command.
     If found, push the handle up in the cache to improve cache hit speed,
     and return it. Otherwise return NULL. The caller should hold
     pocl_dlhandle_lock. */
  POCL_EXPORT
  pocl_dlhandle_cache_item *
  fetch_dlhandle_cache_item (const void *hash, const ulong local_size[3],
                             int specialize, const ulong global_offset[3],
                             const ulong num_groups[3]);

  /* TODO: Refactor the two following functions to not dlopen twice */
  POCL_EXPORT
  void *pocl_get_symbol (const char *symbol_name, const char *dylib_name);

  POCL_EXPORT
  pocl_dlhandle_cache_item *
  pocl_add_cache_item (const void *hash, const ulong local_size[3], int retain,
                       int specialize, const ulong global_offset[3],
                       const ulong num_groups[3], const char *dylib_name,
                       const char *kernel_name);

POCL_EXPORT
void pocl_release_dlhandle_cache (void *dlhandle_cache_item);

POCL_EXPORT
void pocl_setup_device_for_system_memory(cl_device_id device);

POCL_EXPORT
void pocl_reinit_system_memory();

POCL_EXPORT
void pocl_set_buffer_image_limits(cl_device_id device);

POCL_EXPORT
void* pocl_aligned_malloc_global_mem(cl_device_id device, size_t align, size_t size);

POCL_EXPORT
void pocl_free_global_mem(cl_device_id device, void *ptr, size_t size);

void pocl_print_system_memory_stats();

POCL_EXPORT
void pocl_init_default_device_infos (cl_device_id dev,
                                     const char *device_extensions);

POCL_EXPORT
void pocl_setup_opencl_c_with_version (cl_device_id dev, int supports_30);

POCL_EXPORT
void pocl_setup_extensions_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_ils_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_features_with_version (cl_device_id dev);

POCL_EXPORT
void pocl_setup_builtin_kernels_with_version (cl_device_id dev);

#ifdef __cplusplus
}
#endif

#endif
