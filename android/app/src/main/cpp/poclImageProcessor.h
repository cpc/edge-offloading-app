//
// Created by rabijl on 8/22/23.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H

#define LOGTAG "poclaisademo"

#include "stdint.h"

#define CSV_HEADER "frameindex,tag,parameter,value\n"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    NO_COMPRESSION = 1,
    YUV_COMPRESSION = 2,
    JPEG_COMPRESSION = 4,
    JPEG_IMAGE = (1 << 3), // if the input is already a compressed image
} compression_t;

enum {
    ENABLE_PROFILING = (1 << 8)
};

#define CHECK_COMPRESSION_T(inp)                    \
    (NO_COMPRESSION == inp) ||         \
    (YUV_COMPRESSION == inp) ||        \
    (JPEG_COMPRESSION == inp) ||       \
    (JPEG_IMAGE == inp)

// 0 - RGB
// 1 - YUV420 NV21 Android (interleaved U/V)
// 2 - YUV420 (U/V separate)
typedef enum {
    RGB = 0,
    YUV_SEMI_PLANAR,
    YUV_PLANAR
} image_format_t;

/**
 * struct that hold relevant data of jpeg images
 */
struct jpeg_image_data_t {
    uint8_t *data;
    uint64_t capacity;
};

/**
 * struct that holds relevant data of yuv images
 */
struct yuv_image_data_t {
    uint8_t *planes[3];
    int pixel_strides[3];
    int row_strides[3];
};

/**
 * enum of different image types supported by
 * image_data_t
 */
typedef enum {
    YUV_DATA_T = 0,
    JPEG_DATA_T,
} image_datatype_t;

/**
 * pseudopolymorphic struct that can contain different image data
 */
typedef struct {
    image_datatype_t type;
    union {
        struct yuv_image_data_t yuv;
        struct jpeg_image_data_t jpeg;
    } data;
} image_data_t;

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
poclProcessImage(const int device_index, const int do_segment,
                 const compression_t compressionType,
                 const int quality, const int rotation, int32_t *detection_array,
                 uint8_t *segmentation_array, image_data_t image_data, long image_timestamp);

char *
get_c_log_string_pocl();

void
pocl_remote_get_traffic_stats(uint64_t *out_buf, int server_num);

const char *
get_compression_name(const compression_t compression_id);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
