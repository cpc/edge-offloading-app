//
// Created by rabijl on 22.9.2023.
//

#ifndef POCL_AISA_DEMO_ANDROID_MEDIA_CODEC_H
#define POCL_AISA_DEMO_ANDROID_MEDIA_CODEC_H

#include <CL/cl.h>
#include <pocl_types.h>
#include <pocl_cl.h>
#include <stdio.h>
#include "media/NdkMediaCodec.h"
#include "pocl_export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /**
     * the codec used to encode images
     */
    AMediaCodec *codec;
    /**
     * used to set presentation time of the encoder buffer
     */
    uint64_t frame_clock;
    /**
     * the increment size for the frame_clock
     */
    uint64_t time_increment;

    /**
     * the format used to configure the codec
     */
    AMediaFormat *format;

    /**
     * file pointer to write output of encoder to. useful for debugging
     */
    FILE *write_ptr;
} media_codec_state_t;

POCL_EXPORT
void _pocl_kernel_pocl_encode_hevc_yuv420nv21_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void
_pocl_kernel_pocl_configure_hevc_yuv420nv21_workgroup(cl_uchar *args, cl_uchar *context, ulong group_x, ulong group_y,
                                                      ulong group_z);

POCL_EXPORT
void init_mediacodec_encoder(cl_program program, cl_uint device_i);

POCL_EXPORT
void destroy_mediacodec_encoder(cl_device_id device, cl_program program,
                                unsigned dev_i);

void
create_media_codec(media_codec_state_t *state);

void
configure_codec(media_codec_state_t *state, int32_t width, int32_t height, int32_t framerate,
                int32_t i_frame_interval, int32_t bitrate);

void
encode_image(media_codec_state_t *state, const uint8_t *input_buf, uint64_t input_size, uint8_t *output_buf,
             uint64_t output_buf_size, uint64_t *data_written);

void destroy_codec(media_codec_state_t *state);


/***** android software encoder *****/
POCL_EXPORT
void init_c2_android_hevc_encoder(cl_program program, cl_uint device_i);

POCL_EXPORT
void destroy_c2_android_hevc_encoder(cl_device_id device, cl_program program,
                                     unsigned dev_i);

POCL_EXPORT
void
_pocl_kernel_pocl_configure_c2_android_hevc_yuv420nv21_workgroup(cl_uchar *args, cl_uchar *context, ulong group_x, ulong group_y,
                                                                 ulong group_z);

POCL_EXPORT
void
_pocl_kernel_pocl_encode_c2_android_hevc_yuv420nv21_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

void
create_c2_android_hevc_encoder(media_codec_state_t *state);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_ANDROID_MEDIA_CODEC_H
