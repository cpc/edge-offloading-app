//
// Created by rabijl on 16.8.2023.
//

#ifndef _TURBO_JPEG_H_
#define _TURBO_JPEG_H_

// #include <stdlib.h>
#include <stdint.h>
#include <CL/cl.h>
#include <pocl_types.h>
#include <pocl_cl.h>

POCL_EXPORT
void _pocl_kernel_pocl_compress_to_jpeg_yuv420nv21_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_decompress_from_jpeg_rgb888_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

int32_t
turbo_jpeg_run_decompress_from_jpeg_rgb888(const uint8_t *input,
                                           const uint64_t *input_size,
                                           int32_t width,
                                           int32_t height,
                                           uint8_t *output);

int32_t
turbo_jpeg_run_compress_to_jpeg_yuv420nv21(const uint8_t *input,
                                           int32_t width,
                                           int32_t height,
                                           int32_t quality,
                                           uint8_t *output,
                                           uint64_t *output_size);

POCL_EXPORT
void _pocl_kernel_pocl_init_decompress_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_destroy_decompress_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_decompress_from_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

void
turbo_jpeg_run_init_decompress_jpeg_handle_rgb888(uint8_t *const ctx_handle);

void
turbo_jpeg_run_destroy_decompress_jpeg_handle_rgb888(uint8_t *const ctx_handle);

void
turbo_jpeg_run_decompress_from_jpeg_handle_rgb888(const uint8_t *ctx_handle,
                                                  const uint8_t *input,
                                                  const uint64_t *input_size,
                                                  uint8_t *output);

POCL_EXPORT
void init_turbo_jpeg(cl_program program, cl_uint device_i);

POCL_EXPORT
void destroy_turbo_jpeg(cl_device_id device, cl_program program,
                        unsigned dev_i);

#endif //_TURBO_JPEG_H_
