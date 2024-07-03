//
// Created by rabijl on 27.2.2024.
//

#ifndef POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H
#define POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H

//#include "poclImageProcessor.h"
#include "poclImageProcessorTypes.h"
#include "yuv_compression.h"
#include "jpeg_compression.h"
#include "hevc_compression.h"
#include "dnn_stage.h"

#include "testapps.h"
#include <stdint.h>
#include <semaphore.h>
#include <time.h>

#include <Tracy.hpp>
#include <TracyOpenCL.hpp>

#define CSV_HEADER "frame_id,tag,parameter,value\n"
#define MAX_NUM_CL_DEVICES 4

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

#define DETECTION_SIZE     (1 + MAX_DETECTIONS * 6)
#define SEGMENTATION_SIZE  (MAX_DETECTIONS * MASK_W * MASK_H)
#define RECONSTRUCTED_SIZE (MASK_W * MASK_H * 4) // RGBA image
#define TOTAL_OUT_SIZE     (DETECTION_SIZE + SEGMENTATION_SIZE)

#ifdef __cplusplus
extern "C" {
#endif

#define POCL_IMAGE_PROCESSOR_ERROR -100
#define POCL_IMAGE_PROCESSOR_UNRECOVERABLE_ERROR -101

typedef enum {
    LANE_REMOTE_LOST = -3, LANE_SHUTDOWN = -2, LANE_ERROR = -1, LANE_READY = 0, LANE_BUSY = 1,
} lane_state_t;

typedef struct {
    event_array_t *event_array;
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

    char lane_name[16]; // the name shown in tracy

    pthread_mutex_t state_mut;
    /**
     * -2 shutdown
     * -1 error
     * 0 ready
     * 1 busy
     */
    lane_state_t state;
    int local_only; // used to indicate that it is not possible to use the remote device

} pipeline_context;

typedef struct {
    cl_event event_list[2];
    int event_list_size;
} dnn_results;

typedef struct {
    bool is_eval_running;  // tells whether there is an image running currently in the eval pipeline
    pipeline_context *eval_pipeline;
    cl_event dnn_out_event;   // Postprocessing event from enqueue_dnn of the uncompressed eval frame
    cl_event iou_read_event;  // Reading IoU result
    tmp_buf_ctx_t tmp_buf_ctx;
    event_array_t *event_array;
    struct timespec next_eval_ts;
    float iou;
} eval_pipeline_context_t;

typedef struct {
    // unless mentioned, these arrays act like loop buffers
    frame_metadata_t *metadata_array; // metadata on images used for
    pipeline_context *pipeline_array; // collection of pipelines that can run independently from each other
    dnn_results *collected_results; // buffer with processed image results returned with receive_image
    // TODO: add mutex to index head and tail
    int frame_index_head; // index of the array to write data to
    int frame_index_tail; // index of the array to read data from
    int lane_count;
    sem_t pipe_sem; // keep track of how many pipelines are busy
    sem_t local_sem; // local execution can't handle many lanes
    sem_t image_sem; // keep track of how many images are available
    int file_descriptor; // used to log info

    ping_fillbuffer_context_t *ping_context; // used to measure pings
    cl_command_queue remote_queue; // used to run the ping

    cl_command_queue read_queue; // queue to read the collected results

    int devices_found; // used to keep track of how many devices there are

    eval_pipeline_context_t *eval_ctx; // used to run the eval pipeline

    hevc_config_t global_hevc_config; // used to configure the global hevc codec
    int hevc_configured;
    hevc_config_t global_soft_hevc_config; // used to confige the global software hevc codec
    int soft_hevc_configured; //used check that hevc needs to be configured

} pocl_image_processor_context;

int create_pocl_image_processor_context(pocl_image_processor_context **ctx, const int max_lanes,
                                        const int width, const int height, const int config_flags,
                                        const char *codec_sources, const size_t src_size, int fd,
                                        char *service_name);

int dequeue_spot(pocl_image_processor_context *const ctx, const int timeout,
                 const device_type_enum dev_type);

cl_int submit_image_to_pipeline(pipeline_context *ctx, const codec_config_t config,
                                const bool do_reconstruct, const image_data_t image_data,
                                frame_metadata_t *metadata, dnn_results *output,
                                tmp_buf_ctx_t *tmp_buf_ctx);

int submit_image(pocl_image_processor_context *ctx, codec_config_t codec_config,
                 image_data_t image_data, int is_eval_frame, int *frame_index);

int wait_image_available(pocl_image_processor_context *ctx, int timeout);

int receive_image(pocl_image_processor_context *ctx, int32_t *detection_array,
                  uint8_t *segmentation_array, frame_metadata_t *eval_metadata, int *segmentation,
                  collected_events_t *collected_events);

int destroy_pocl_image_processor_context(pocl_image_processor_context **ctx);

float get_last_iou(pocl_image_processor_context *ctx);

int check_and_configure_global_hevc(pocl_image_processor_context *ctx, codec_config_t codec,
                                    pipeline_context *pipeline);

void halt_lanes(pocl_image_processor_context *ctx);

void resume_lanes(pocl_image_processor_context *ctx);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_POCLIMAGEPROCESSORV2_H
