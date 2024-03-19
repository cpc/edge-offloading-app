//
// Created by rabijl on 27.2.2024.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H

#include "poclImageProcessor.h"
#include "yuv_compression.h"
#include "testapps.h"
#include "dnn_stage.h"
#include <stdint.h>
#include <semaphore.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * a stuct that contains metadata needed
 * for evaluation on the reading side
 */
typedef struct {
    int frame_index;
    long image_timestamp;
    int is_eval_frame;
    int segmentation;
    host_ts_ns_t host_ts_ns;
} image_metadata_t;

typedef struct {
    event_array_t event_array;
    int config_flags; // used to configure codecs
    cl_command_queue *enq_queues; // collection of device queues
    int queue_count;
    cl_mem inp_yuv_mem; // buffer to write the inp yuv image to
    cl_uchar *host_inp_buf; // where to copy the image to
    size_t host_inp_buf_size;
    cl_mem comp_to_dnn_buf;

    // different codec options
    yuv_codec_context_t *yuv_context;
#ifndef DISABLE_JPEG
    jpeg_codec_context_t *jpeg_context;
#endif // DISABLE_JPEG
#ifndef DISABLE_HEVC
    hevc_codec_context_t *hevc_context;
    hevc_codec_context_t *software_hevc_context;
#endif // DISABLE_HEVC
    dnn_context_t *dnn_context;

    int width;
    int height;

} pipeline_context;

typedef struct {
    cl_mem detection_array;
    cl_mem segmentation_array;
    cl_event result_event;
} dnn_results;

typedef struct {
    event_array_t *event_array;
    event_array_t *eval_event_array;
    float iou;
    uint64_t size_bytes;
    host_ts_ns_t host_ts_ns;
} eval_metadata_t;

//typedef struct {
//    int config_flags;
//    cl_mem input_buf;
//    event_array_t  event_array;
//    cl_command_queue enq_queue;
//    cl_uchar * host_inp_buf;
//    dnn_context_t *dnn_context;
//}local_pipeline_context_t;

typedef struct {
    // unless mentioned, these arrays act like loop buffers
    image_metadata_t *metadata_array; // metadata on images used for
    pipeline_context *pipeline_array; // collection of pipelines that can run independently from each other
    dnn_results *collected_results; // buffer with processed image results returned with receive_image
    // TODO: add mutex to index head and tail
    int frame_index_head; // index of the array to write data to
    int frame_index_tail; // index of the array to read data from
    int lane_count;
    sem_t pipe_sem; // keep track of how many pipelines are busy
    sem_t image_sem; // keep track of how many images are available
    int file_descriptor; // used to log info

    ping_fillbuffer_context_t *PING_CTX; // used to measure pings

    cl_command_queue read_queue; // queue to read the collected results
} pocl_image_processor_context;

int
create_pocl_image_processor_context(pocl_image_processor_context **ctx, const int max_lanes,
                                    const int width, const int height, const int config_flags,
                                    const char *codec_sources, const size_t src_size, int fd);

int
dequeue_spot(pocl_image_processor_context *ctx, int timeout);

int
submit_image(pocl_image_processor_context *ctx, codec_config_t codec_config,
             image_data_t image_data, int is_eval_frame);

int
wait_image_available(pocl_image_processor_context *ctx, int timeout);

int
receive_image(pocl_image_processor_context *ctx, int32_t *detection_array,
              uint8_t *segmentation_array,
              eval_metadata_t *eval_metadata, int *segmentation);

int
destroy_pocl_image_processor_context(pocl_image_processor_context **ctx);

float
get_last_iou(pocl_image_processor_context *ctx);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H
