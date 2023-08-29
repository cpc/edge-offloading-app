//
// Created by rabijl on 8/22/23.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H

#include "stdint.h"

#define LOGTAG "poclimageprocessor"

#define CSV_HEADER "frameindex,tag,parameter,value\n"

#ifdef __cplusplus
extern "C" {
#endif

int
initPoclImageProcessor(const int width, const int height, const bool enableProfiling,
                       const char *codec_sources, const size_t src_size, const int fd);

int
destroy_pocl_image_processor();

int
poclProcessYUVImage(const int device_index, const int do_segment, const int do_compression,
                    const int rotation, const int8_t *y_ptr, const int yrow_stride,
                    const int ypixel_stride, const int8_t *u_ptr, const int8_t *v_ptr,
                    const int uvrow_stride, const int uvpixel_stride, int32_t *detection_array,
                    int8_t *segmentation_array);

char *
get_c_log_string_pocl();

void
pocl_remote_get_traffic_stats(uint64_t *out_buf, int server_num);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
