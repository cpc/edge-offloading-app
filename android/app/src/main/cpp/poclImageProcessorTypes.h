//
// Created by rabijl on 19.3.2024.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSORTYPES_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSORTYPES_H

#include "hevc_compression.h"
#include "jpeg_compression.h"

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
    NO_COMPRESSION = 1,
    YUV_COMPRESSION = 2,
    JPEG_COMPRESSION = 4,
    JPEG_IMAGE = (1 << 3), // if the input is already a compressed image
    HEVC_COMPRESSION = (1 << 4),
    SOFTWARE_HEVC_COMPRESSION = (1 << 5),
    // when adding new compression types
    // make to stay below the ENABLE_PROFILING ENUM
    // and also add similar entries on the java side
} compression_t;

typedef enum {
    LOCAL_DEVICE = 0,
    PASSTHRU_DEVICE = 1,
    REMOTE_DEVICE = 2,
    REMOTE_DEVICE_2 = 3,
} device_type_enum;

enum {
    ENABLE_PROFILING = (1 << 8),
    LOCAL_ONLY = (1 << 9)
};

/**
 * Host device timestamps (in nanoseconds) in the image processing loop. Should be ordered.
 */
typedef struct {
    int64_t start;
    int64_t before_enc;
    int64_t before_fill;
    int64_t before_dnn;
    int64_t before_eval;
    int64_t before_wait;
    int64_t after_wait;
    int64_t stop;
    int64_t fill_ping_duration;
} host_ts_ns_t;

int supports_config_flags(const int input);

#define CHECK_COMPRESSION_T(inp)                    \
    (NO_COMPRESSION == inp) ||         \
    (YUV_COMPRESSION == inp) ||        \
    (JPEG_COMPRESSION == inp) ||       \
    (JPEG_IMAGE == inp)  ||                         \
    (HEVC_COMPRESSION == inp) ||                      \
    (SOFTWARE_HEVC_COMPRESSION == inp)

// 0 - RGB
// 1 - YUV420 NV21 Android (interleaved U/V)
// 2 - YUV420 (U/V separate)
typedef enum {
    RGB = 0,
    YUV_SEMI_PLANAR,
    YUV_PLANAR
} pixel_format_enum;

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
    long image_timestamp;
} image_data_t;

/**
 * A struct that contains configuration parameters required to
 * process an image with the pocl image processor
 */
typedef struct {
    compression_t compression_type; // codec to be used
    device_type_enum device_type; // device on which the image is processed
    int rotation;   // indicate if the image needs to be rotated
    int do_segment; // indicate that the segmentation needs to also be run
    union {
        jpeg_config_t jpeg;
        hevc_config_t hevc;
    } config; // codec specific configuration parameters
    int id; // ID of the codec config pointing at the array of available configs
} codec_config_t;

/**
 * a stuct that contains metadata needed
 * for evaluation on the reading side
 */
//typedef struct {
//    int frame_index;
//    long image_timestamp;
//    int is_eval_frame;
//    int segmentation;
//    host_ts_ns_t host_ts_ns;
//} image_metadata_t;

typedef struct {
    event_array_t *event_array;
    event_array_t *eval_event_array;
    float iou;
    uint64_t size_bytes;
    host_ts_ns_t host_ts_ns;
    int frame_index;
    long image_timestamp;
    int is_eval_frame;
    codec_config_t codec;
} frame_metadata_t;

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSORTYPES_H
