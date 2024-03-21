//
// Created by rabijl on 19.3.2024.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSORUTILS_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSORUTILS_H

#include "poclImageProcessorTypes.h"

#ifdef __cplusplus
extern "C" {
#endif

const char *
get_compression_name(const compression_t compression_id);

void
copy_yuv_to_arrayV2(const int width, const int height, const image_data_t image,
                    const compression_t compression_type,
                    cl_uchar *const dest_buf);

void
log_eval_metadata(const int fd, const int frame_index, const eval_metadata_t metadata);

void
log_codec_config(const int fd, const int frame_index, const codec_config_t config);

void
log_host_ts_ns(const int fd, const int frame_index, const host_ts_ns_t host_ts);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSORUTILS_H
