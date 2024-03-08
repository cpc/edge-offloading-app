//
// Created by rabijl on 8/22/23.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSOR_H

#define LOGTAG "poclaisademo"

#include "stdint.h"
#include "event_logger.h"
#include "hevc_compression.h"
#include "jpeg_compression.h"

#define CSV_HEADER "frameindex,tag,parameter,value\n"
#define MAX_NUM_CL_DEVICES 4

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define DETECTION_SIZE     (1 + MAX_DETECTIONS * 6)
#define SEGMENTATION_SIZE  (MAX_DETECTIONS * MASK_W * MASK_H)
#define RECONSTRUCTED_SIZE (MASK_W * MASK_H * 4) // RGBA image
#define TOTAL_OUT_SIZE     (DETECTION_SIZE + SEGMENTATION_SIZE)
#define TOTAL_RX_SIZE      (DETECTION_SIZE + MASK_W * MASK_H)
#define ANDROID_MTU        1464  // found via ifconfig
#define SERVER_MTU         1500  // found via ifconfig
//#define FILL_PING_SIZE     (ANDROID_MTU + SERVER_MTU)
#define FILL_PING_SIZE     (73 + 120) // write + read size read from POCL_DEBUG=remote
//#define FILL_PING_SIZE     1  // buffer size used on the host side

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
}devic_type_enum;

enum {
    ENABLE_PROFILING = (1 << 8)
};

/**
 * Host device timestamps (in nanoseconds) in the image processing loop. Should be ordered.
 */
typedef struct {
    int64_t start;
    int64_t before_fill;
    int64_t before_enc;
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
    devic_type_enum device_type; // device on which the image is processed
    int rotation;   // indicate if the image needs to be rotated
    int do_segment; // indicate that the segmentation needs to also be run
    union {
        jpeg_config_t jpeg;
        hevc_config_t hevc;
    } config; // codec specific configuration parameters
} codec_config_t;

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
                       const char *codec_sources, const size_t src_size, int fd,
                       event_array_t *event_array, event_array_t *eval_event_array);

int
destroy_pocl_image_processor();

//int64_t get_timestamp_ns();

int
poclProcessImage(codec_config_t condec_config, int frame_index, int do_segment,
                 int is_eval_frame, int rotation, int32_t *detection_array,
                 uint8_t *segmentation_array, event_array_t *event_array,
                 event_array_t *eval_event_array, image_data_t image_data, long image_timestamp,
                 float *iou, uint64_t *size_bytes, host_ts_ns_t *host_ts_ns);
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
