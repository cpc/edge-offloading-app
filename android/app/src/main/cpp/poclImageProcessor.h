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

//#define NO_COMPRESSION 0
//#define YUV_COMPRESSION 1
//#define JPEG_COMPRESSION 2

typedef enum {
    NO_COMPRESSION = 1,
    YUV_COMPRESSION = 2,
    JPEG_COMPRESSION = 4
} compression_t;

enum {
    ENABLE_PROFILING = (1 << 8)
};

#define CHECK_COMPRESSION_T(inp)                    \
    (NO_COMPRESSION == compression_type) ||         \
    (YUV_COMPRESSION == compression_type) ||        \
    (JPEG_COMPRESSION == compression_type)

// 0 - RGB
// 1 - YUV420 NV21 Android (interleaved U/V)
// 2 - YUV420 (U/V separate)
typedef enum {
    RGB = 0,
    YUV_SEMI_PLANAR,
    YUV_PLANAR
} image_format_t;

/**
 *
 * @param width
 * @param height
 * @param enableProfiling
 * @param config_flags flags that can be used to enable configurations
 * such as available compression types and input types
 * @param codec_sources
 * @param src_size
 * @param fd
 * @return
 */
int
initPoclImageProcessor(const int width, const int height, const int config_flags,
                       const char *codec_sources, const size_t src_size, const int fd);

int
destroy_pocl_image_processor();

int
poclProcessYUVImage(const int device_index, const int do_segment, const compression_t compression,
                    const int quality, const int rotation, const uint8_t *y_ptr,
                    const int yrow_stride,
                    const int ypixel_stride, const uint8_t *u_ptr, const uint8_t *v_ptr,
                    const int uvrow_stride, const int uvpixel_stride, int32_t *detection_array,
                    uint8_t *segmentation_array);

char *
get_c_log_string_pocl();

void
pocl_remote_get_traffic_stats(uint64_t *out_buf, int server_num);

char *
get_compression_name(const compression_t compression_id);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
