//
// Created by rabijl on 19.3.2024.
//

#include "poclImageProcessorUtils.h"
#include <assert.h>
#include <Tracy.hpp>

#ifdef __cplusplus
extern "C" {
#endif

const char *
get_compression_name(const compression_t compression_id) {
    switch (compression_id) {
        case NO_COMPRESSION:
            return "none";
        case YUV_COMPRESSION:
            return "yuv";
        case JPEG_COMPRESSION:
            return "jpeg";
        case HEVC_COMPRESSION:
            return "hevc";
        case SOFTWARE_HEVC_COMPRESSION:
            return "software_hevc";
        case JPEG_IMAGE:
            return "jpeg_image";
        default:
            return "unknown";
    }
}

/**
 * function to copy raw buffers from the image to a local array and make sure the result is in
 * nv21 format.
 * @param width
 * @param height
 * @param image needs to be a yuv image
 * @param compression_type
 * @param dest_buf
 */
void
copy_yuv_to_arrayV2(const int width, const int height, const image_data_t image,
                    const compression_t compression_type,
                    cl_uchar *const dest_buf) {
  ZoneScoped;
    assert(image.type == YUV_DATA_T && "image is not a yuv image");

    // this will be optimized out by the compiler
    const int yrow_stride = image.data.yuv.row_strides[0];
    const uint8_t *y_ptr = image.data.yuv.planes[0];
    const uint8_t *u_ptr = image.data.yuv.planes[1];
    const uint8_t *v_ptr = image.data.yuv.planes[2];
    const int ypixel_stride = image.data.yuv.pixel_strides[0];
    const int upixel_stride = image.data.yuv.pixel_strides[1];
    const int vpixel_stride = image.data.yuv.pixel_strides[2];

    // copy y plane into buffer
    for (int i = 0; i < height; i++) {
        // row_stride is in bytes
        for (int j = 0; j < yrow_stride; j++) {
            dest_buf[i * yrow_stride + j] = y_ptr[(i * yrow_stride + j) * ypixel_stride];
        }
    }

    int uv_start_index = height * yrow_stride;
    // interleave u and v regardless of if planar or semiplanar
    // divided by 4 since u and v are subsampled by 2
    for (int i = 0; i < (height * width) / 4; i++) {
        if (HEVC_COMPRESSION == compression_type) {
            dest_buf[uv_start_index + 1 + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 2 * i] = u_ptr[i * upixel_stride];
        } else {
            dest_buf[uv_start_index + 2 * i] = v_ptr[i * vpixel_stride];
            dest_buf[uv_start_index + 1 + 2 * i] = u_ptr[i * upixel_stride];
        }
    }

}

void
log_eval_metadata(const int file_descriptor, const int frame_index,
                   const frame_metadata_t metadata) {
    dprintf(file_descriptor, "%d,frame,timestamp,%ld\n", frame_index,
            metadata.image_timestamp);
    dprintf(file_descriptor, "%d,frame,is_eval,%d\n", frame_index, metadata.is_eval_frame);
//    dprintf(file_descriptor, "%d,config,segment,%d\n", frame_index, metadata.segmentation);
    log_host_ts_ns(file_descriptor, frame_index, metadata.host_ts_ns);
}

void
log_host_ts_ns(const int file_descriptor, const int frame_index, const host_ts_ns_t host_ts) {
    dprintf(file_descriptor, "%d,frame_time,start_ns,%lu\n", frame_index, host_ts.start);
    dprintf(file_descriptor, "%d,frame_time,before_enc_ns,%lu\n", frame_index, host_ts.before_enc);
    dprintf(file_descriptor, "%d,frame_time,before_fill_ns,%lu\n", frame_index,
            host_ts.before_fill);
    dprintf(file_descriptor, "%d,frame_time,before_dnn_ns,%lu\n", frame_index, host_ts.before_dnn);
    dprintf(file_descriptor, "%d,frame_time,before_eval_ns,%lu\n", frame_index,
            host_ts.before_eval);
    dprintf(file_descriptor, "%d,frame_time,before_wait_ns,%lu\n", frame_index,
            host_ts.before_wait);
    dprintf(file_descriptor, "%d,frame_time,after_wait_ns,%lu\n", frame_index, host_ts.after_wait);
    dprintf(file_descriptor, "%d,frame_time,stop_ns,%lu\n", frame_index, host_ts.stop);
}

void
log_codec_config(const int file_descriptor, const int frame_index, const codec_config_t config) {

    dprintf(file_descriptor, "%d,device,index,%d\n", frame_index, config.device_type);
    dprintf(file_descriptor, "%d,config,segment,%d\n", frame_index, config.do_segment);
    dprintf(file_descriptor, "%d,compression,name,%s\n", frame_index,
            get_compression_name(config.compression_type));

    // depending on the codec config log different parameters
    if (JPEG_COMPRESSION == config.compression_type) {
        dprintf(file_descriptor, "%d,compression,quality,%d\n", frame_index,
                config.config.jpeg.quality);
    } else if (HEVC_COMPRESSION == config.compression_type) {
        dprintf(file_descriptor, "%d,compression,i_frame_interval,%d\n", frame_index,
                config.config.hevc.i_frame_interval);
        dprintf(file_descriptor, "%d,compression,framerate,%d\n", frame_index,
                config.config.hevc.framerate);
        dprintf(file_descriptor, "%d,compression,bitrate,%d\n", frame_index,
                config.config.hevc.bitrate);
    }

}

#ifdef __cplusplus
}
#endif
