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
#include "poclImageProcessorTypes.h"

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
