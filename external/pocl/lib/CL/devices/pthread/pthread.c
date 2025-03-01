/* OpenCL native pthreaded device implementation.

   Copyright (c) 2011-2012 Universidad Rey Juan Carlos
                 2011-2019 Pekka Jääskeläinen

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

#define _GNU_SOURCE
#define __USE_GNU

#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#include "builtin_kernels.hh"
#include "builtin-kernels/metadata.h"

#ifndef _WIN32
#  include <unistd.h>
#else
#  include "vccompat.hpp"
#endif

#include "common.h"
#include "common_utils.h"
#include "config.h"
#include "devices.h"
#include "pocl-pthread.h"
#include "pocl-pthread_scheduler.h"
#include "pocl_mem_management.h"
#include "pocl_util.h"

#ifdef ENABLE_LLVM
#include "pocl_llvm.h"
#endif

/**
 * Per event data.
 */
struct event_data {
  pthread_cond_t event_cond;
};


static pocl_lock_t pocl_dlhandle_lock;
typedef void (*init_func_t) (cl_program program, cl_uint device_i);
typedef void (*free_func_t) (cl_device_id device, cl_program program,
                             unsigned dev_i);

void
pocl_pthread_init_device_ops(struct pocl_device_ops *ops)
{
  pocl_basic_init_device_ops(ops);

  ops->device_name = "cpu";

  /* implementation that differs from basic */
  ops->probe = pocl_pthread_probe;
  ops->uninit = pocl_pthread_uninit;
  ops->reinit = pocl_pthread_reinit;
  ops->init = pocl_pthread_init;
  ops->run = pocl_pthread_run;
  ops->join = pocl_pthread_join;
  ops->submit = pocl_pthread_submit;
  ops->notify = pocl_pthread_notify;
  ops->broadcast = pocl_broadcast;
  ops->flush = pocl_pthread_flush;
  ops->wait_event = pocl_pthread_wait_event;
  ops->notify_event_finished = pocl_pthread_notify_event_finished;
  ops->notify_cmdq_finished = pocl_pthread_notify_cmdq_finished;
  ops->update_event = pocl_pthread_update_event;
  ops->free_event_data = pocl_pthread_free_event_data;

  ops->init_queue = pocl_pthread_init_queue;
  ops->free_queue = pocl_pthread_free_queue;

  ops->post_build_program = pocl_pthread_add_host_builtins;
  ops->free_program = pocl_pthread_free_program;

#ifdef ENABLE_OPENCV_ONNX
  // if the onnx builtin kernels are loaded, startup onnx when the device
  // starts up and worry about freeing it up later
  void (*onnx_fn)(void) = pocl_get_symbol("kick_onnx_awake", "libpocl_pthread_opencv_onnx.so");
  if(NULL != onnx_fn){
      onnx_fn();
  }
#endif

#ifdef ENABLE_FFMPEG_DECODE
  void (* ffmpeg_init)(void) = pocl_get_symbol("kick_ffmpeg_awake", "libpocl_pthread_ffmpeg_decoder.so");
  if(NULL != ffmpeg_init){
      ffmpeg_init();
  }
#endif

}

unsigned int
pocl_pthread_probe (struct pocl_device_ops *ops)
{
  int env_count = pocl_device_get_env_count(ops->device_name);

  /* for backwards compatibility */
  if (env_count <= 0)
    env_count = pocl_device_get_env_count("pthread");

  /* Env was not specified, default behavior was to use 1 pthread device
   * unless tbb device is being built. */
  if (env_count < 0)
#ifdef BUILD_TBB
    return 0;
#else
    return 1;
#endif

  return env_count;
}

static cl_device_partition_property pthread_partition_properties[2]
    = { CL_DEVICE_PARTITION_EQUALLY, CL_DEVICE_PARTITION_BY_COUNTS };
static int scheduler_initialized = 0;

static cl_bool pthread_available = CL_TRUE;
static cl_bool pthread_unavailable = CL_FALSE;

cl_int
pocl_pthread_init (unsigned j, cl_device_id device, const char* parameters)
{
  int err;

  device->data = NULL;
  device->available = &pthread_unavailable;

  cl_int ret = pocl_cpu_init_common (device);
  if (ret != CL_SUCCESS)
    return ret;

  pocl_init_dlhandle_cache ();
  pocl_init_kernel_run_command_manager ();

  /* pthread has elementary partitioning support,
   * but only if OpenMP is disabled */
#ifdef ENABLE_HOST_CPU_DEVICES_OPENMP
  device->max_sub_devices = 0;
  device->num_partition_properties = 0;
  device->num_partition_types = 0;
  device->partition_type = NULL;
  device->partition_properties = NULL;
#else
  device->max_sub_devices = device->max_compute_units;
  device->num_partition_properties = 2;
  device->partition_properties = pthread_partition_properties;
  device->num_partition_types = 0;
  device->partition_type = NULL;
#endif

  device->builtin_kernel_list = "pocl.add.i8;"
                                "pocl.dnn.detection.u8;"
                                "pocl.dnn.segmentation.postprocess.u8;"
                                "pocl.dnn.segmentation.reconstruct.u8;"
                                "pocl.dnn.eval.iou.f32;"
                                "pocl.decompress.from.jpeg.rgb888;"
                                "pocl.compress.to.jpeg.yuv420nv21;"
                                "pocl.encode.hevc.yuv420nv21;"
                                "pocl.decode.hevc.yuv420nv21;"
                                "pocl.configure.hevc.yuv420nv21;"
                                "pocl.encode.c2.android.hevc.yuv420nv21;"
                                "pocl.configure.c2.android.hevc.yuv420nv21;"
                                "pocl.init.decompress.jpeg.handle.rgb888;"
                                "pocl.decompress.from.jpeg.handle.rgb888;"
                                "pocl.destroy.decompress.jpeg.handle.rgb888;"
                                "pocl.dnn.ctx.init;"
                                "pocl.dnn.ctx.destroy;"
                                "pocl.dnn.ctx.detection.u8;"
                                "pocl.dnn.ctx.segmentation.postprocess.u8;"
                                "pocl.dnn.ctx.segmentation.reconstruct.u8;"
                                "pocl.dnn.ctx.eval.iou.f32";
  // device->builtin_kernel_list = "pocl.add.i8";
    device->num_builtin_kernels = 21;

  if (!scheduler_initialized)
    {
      ret = pthread_scheduler_init (device);
      if (ret == CL_SUCCESS)
        {
          scheduler_initialized = 1;
        }
    }

  device->available = &pthread_available;

  return ret;
}

cl_int
pocl_pthread_uninit (unsigned j, cl_device_id device)
{
  if (scheduler_initialized)
    {
      pthread_scheduler_uninit (device);
      scheduler_initialized = 0;
    }

  POCL_MEM_FREE (device->data);
  return CL_SUCCESS;
}

cl_int
pocl_pthread_reinit (unsigned j, cl_device_id device, const char *parameters)
{
  cl_int ret = CL_SUCCESS;

  if (!scheduler_initialized)
    {
      ret = pthread_scheduler_init (device);
      if (ret == CL_SUCCESS)
        {
          scheduler_initialized = 1;
        }
    }

  return ret;
}

void
pocl_pthread_run (void *data, _cl_command_node *cmd)
{
  /* not used: this device will not be told when or what to run */
}

void
pocl_pthread_submit (_cl_command_node *node, cl_command_queue cq)
{
  node->node_state = COMMAND_READY;
  if (pocl_command_is_ready (node->sync.event.event))
    {
      pocl_update_event_submitted (node->sync.event.event);
      pthread_scheduler_push_command (node);
    }
  POCL_UNLOCK_OBJ (node->sync.event.event);
  return;
}

void
pocl_pthread_flush(cl_device_id device, cl_command_queue cq)
{

}

void
pocl_pthread_join(cl_device_id device, cl_command_queue cq)
{
  POCL_LOCK_OBJ (cq);
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  while (1)
    {
      if (cq->command_count == 0)
        {
          POCL_UNLOCK_OBJ (cq);
          return;
        }
      else
        {
          PTHREAD_CHECK (pthread_cond_wait (cq_cond, &cq->pocl_lock));
        }
    }
  return;
}

void
pocl_pthread_notify (cl_device_id device, cl_event event, cl_event finished)
{
  _cl_command_node *node = event->command;

  if (finished->status < CL_COMPLETE)
    {
      pocl_update_event_failed (event);
      return;
    }

  if (node->node_state != COMMAND_READY)
    {
      POCL_MSG_PRINT_EVENTS (
          "pthread: command related to the notified event %lu not ready\n",
          event->id);
      return;
    }

  if (pocl_command_is_ready (node->sync.event.event))
    {
      if (event->status == CL_QUEUED)
        {
          pocl_update_event_submitted (event);
          pthread_scheduler_push_command (node);
        }
    }

  return;
}

void
pocl_pthread_notify_cmdq_finished (cl_command_queue cq)
{
  /* must be called with CQ already locked.
   * this must be a broadcast since there could be multiple
   * user threads waiting on the same command queue
   * in pthread_scheduler_wait_cq(). */
  pthread_cond_t *cq_cond = (pthread_cond_t *)cq->data;
  PTHREAD_CHECK (pthread_cond_broadcast (cq_cond));
}

void
pocl_pthread_notify_event_finished (cl_event event)
{
  struct event_data *e_d = event->data;
  PTHREAD_CHECK (pthread_cond_broadcast (&e_d->event_cond));
}

void
pocl_pthread_update_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = NULL;
  if (event->data == NULL && event->status == CL_QUEUED)
    {
      e_d = malloc(sizeof(struct event_data));
      assert(e_d);

      PTHREAD_CHECK (pthread_cond_init (&e_d->event_cond, NULL));
      event->data = (void *) e_d;

      VG_ASSOC_COND_VAR (e_d->event_cond, event->pocl_lock);
    }
}

void pocl_pthread_wait_event (cl_device_id device, cl_event event)
{
  struct event_data *e_d = event->data;

  POCL_LOCK_OBJ (event);
  while (event->status > CL_COMPLETE)
    {
      PTHREAD_CHECK (pthread_cond_wait (&e_d->event_cond, &event->pocl_lock));
    }
  POCL_UNLOCK_OBJ (event);
}


void pocl_pthread_free_event_data (cl_event event)
{
  assert(event->data != NULL);
  free(event->data);
  event->data = NULL;
}

int
pocl_pthread_init_queue (cl_device_id device, cl_command_queue queue)
{
  queue->data
      = pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, sizeof (pthread_cond_t));
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_init (cond, NULL));

  POCL_LOCK_OBJ (queue);
  VG_ASSOC_COND_VAR ((*cond), queue->pocl_lock);
  POCL_UNLOCK_OBJ (queue);

  return CL_SUCCESS;
}

int
pocl_pthread_free_queue (cl_device_id device, cl_command_queue queue)
{
  pthread_cond_t *cond = (pthread_cond_t *)queue->data;
  PTHREAD_CHECK (pthread_cond_destroy (cond));
  POCL_MEM_FREE (queue->data);
  return CL_SUCCESS;
}

int
pocl_pthread_add_host_builtins (cl_program program, cl_uint device_i)
{
    // TODO: Is it possible to set these dynamically?
    const ulong local_size[3] = { 1, 1, 1 };
    const ulong global_offset[3] = { 0, 0, 0 };
    const ulong num_groups[3] = { 1, 1, 1 };
    const int retain = 1;
    const int specialize = 1;

    for (int i = 0; i < NUM_PTHREAD_BUILTIN_HOST_KERNELS; ++i)
    {
        char *kernel_name = kernel_names[i];
        const char *dylib_name = dylib_names[i];
        const char *init_fn_name = init_fn_names[i];

        for (int j = 0; j < program->num_kernels; ++j)
        {
            const pocl_kernel_metadata_t *kernel_meta = &program->kernel_meta[j];

            // Skip program kernels that are not host builtin kernels
            if (strcmp (kernel_names[i], kernel_meta->name) != 0)
            {
                continue;
            }

            // For each host builtin kernel call its init function
            if (strlen(init_fn_name) > 0)
            {
                const init_func_t init_fn = pocl_get_symbol(init_fn_name, dylib_name);
                init_fn(program, device_i);
            }

            const pocl_kernel_hash_t *hash = kernel_meta->build_hash;
            char *kernel_meta_name = kernel_meta->name;

            POCL_LOCK (pocl_dlhandle_lock);
            pocl_dlhandle_cache_item *ci = fetch_dlhandle_cache_item (
                    (const void *)(hash), local_size, specialize, global_offset,
                    num_groups);

            if (ci != NULL)
            {
                if (retain)
                    ++ci->ref_count;
                POCL_UNLOCK (pocl_dlhandle_lock);
                return 0;
            }

            char *saved_name = NULL;
            pocl_sanitize_builtin_kernel_name2 (&kernel_name, &kernel_meta_name,
                                                NUM_PTHREAD_BUILTIN_HOST_KERNELS,
                                                &saved_name);
            ci = pocl_add_cache_item (hash, local_size, retain, specialize,
                                      global_offset, num_groups, dylib_name,
                                      kernel_name);
            pocl_restore_builtin_kernel_name2 (&kernel_name, &kernel_meta_name,
                                               NUM_PTHREAD_BUILTIN_HOST_KERNELS,
                                               &saved_name);

            POCL_UNLOCK (pocl_dlhandle_lock);

        }
    }

    return 0;
}

int
pocl_pthread_free_program (cl_device_id device, cl_program program,
                          unsigned dev_i)
{

    // For each host builtin kernel call its free function
    for (int i = 0; i < NUM_PTHREAD_BUILTIN_HOST_KERNELS; ++i)
    {
        const char *dylib_name = dylib_names[i];
        const char *free_fn_name = free_fn_names[i];

        for (int j = 0; j < program->num_kernels; ++j)
        {
            const pocl_kernel_metadata_t *kernel_meta = &program->kernel_meta[j];

            // Skip program kernels that are not host builtin kernels
            if (strcmp (kernel_names[i], kernel_meta->name) != 0)
            {
                continue;
            }

            if (strlen(free_fn_name) > 0)
            {
                const free_func_t free_fn = pocl_get_symbol(free_fn_name, dylib_name);
                free_fn(device, program, dev_i);
            }
        }
    }

#ifdef ENABLE_LLVM
    pocl_llvm_free_llvm_irs (program, dev_i);
#endif
    return 0;
}
