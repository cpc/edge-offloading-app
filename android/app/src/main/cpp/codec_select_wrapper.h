//
// Created by rabijl on 1.8.2024.
//

#ifndef POCL_AISA_DEMO_CODEC_SELECT_WRAPPER_H
#define POCL_AISA_DEMO_CODEC_SELECT_WRAPPER_H

#include "sharedUtils.h"
#include "eval.h"
#include <cassert>
#include "codec_select.h"
#include "poclImageProcessorV2.h"
#include "poclImageProcessorTypes.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Set codec config from the codec selection state and input parameters from the UI (rotation, do_segment)
 */
inline
static void get_codec_config(codec_select_state_t *state, int rotation, int do_segment,
                             codec_config_t *codec_config) {
    assert(NULL != state);
    const codec_params_t params = get_codec_params(state);
    const int codec_id = get_codec_id(state);
    codec_config->compression_type = params.compression_type;
    codec_config->device_type = params.device_type;
    codec_config->rotation = rotation;
    codec_config->do_segment = do_segment;
    codec_config->id = codec_id;
    if (codec_config->compression_type == JPEG_COMPRESSION) {
        codec_config->config.jpeg = params.config.jpeg;
    } else if (codec_config->compression_type == HEVC_COMPRESSION ||
               codec_config->compression_type == SOFTWARE_HEVC_COMPRESSION) {
        codec_config->config.hevc = params.config.hevc;
    }
}

/**
 * Wrap the pip ctx so that the codec select can analyze and determine the next configuration
 * @param state
 * @param ctx
 * @param device_index
 * @param do_segment
 * @param compression_type
 * @param quality
 * @param rotation
 * @param do_algorithm
 * @param image_data
 * @return OpenCL status
 */
inline static int
codec_select_submit_image(codec_select_state_t *state, pocl_image_processor_context *ctx,
                          int device_index, int do_segment, int compression_type, int quality,
                          int rotation,
                          int do_algorithm, image_data_t *image_data) {

    int status;
    int is_eval_frame = 0;
    int frame_index = -1;

    // read the current config from the codec selection state
    codec_config_t codec_config;

    if (do_algorithm) {
        get_codec_config(state, rotation, do_segment, &codec_config);
    } else {
        // override the codec config with whatever the user set in the UI
        select_codec_manual((device_type_enum) (device_index), do_segment,
                            (compression_t) (compression_type), quality, rotation, &codec_config);
    }

    // check if we need to submit image to the eval pipeline
    if (ctx->enable_eval || state->is_calibrating) {
        // TODO: see if this is taking a lot of time
        status = check_eval(ctx->eval_ctx, state, codec_config, &is_eval_frame);
        CHECK_AND_RETURN(status, "could not check and submit eval frame");
    }

    // submit the image for the actual encoding (needs to be submitted *before* the eval frame)
    bool codec_selected = drain_codec_selected(state);
    int64_t latency_offset_ms = get_latency_offset_ms(state);
    status = submit_image(ctx, codec_config, *image_data, is_eval_frame, codec_selected,
                          latency_offset_ms, &frame_index);
    CHECK_AND_RETURN(status, "could not submit frame");

    if (is_eval_frame) {
        // submit the eval frame if appropriate
        status = run_eval(ctx->eval_ctx, state, codec_config, frame_index, *image_data);
        CHECK_AND_RETURN(status, "could not submit eval frame");
    }

    return status;

}

inline static int
codec_select_receive_image(codec_select_state_t *state, pocl_image_processor_context *ctx,
                           int32_t *detection_array, uint8_t *segmentation_array, int64_t *metadata_array) {

    int status;
    frame_metadata_t metadata;
    //    lane_state_t new_state;
    status = receive_image(ctx, detection_array, segmentation_array, &metadata,
                           (int32_t *) metadata_array, state->collected_events);

    if (status == CL_SUCCESS) {
        // log statistics to codec selection data
        update_stats(&metadata, ctx->eval_ctx, state);
    }

    // not strictly necessary, just easier to debug without having old values lying around
    reset_collected_events(state->collected_events);

    // narrow down to micro seconds
    metadata_array[1] = ((metadata.host_ts_ns.stop - metadata.host_ts_ns.start) / 1000);

    return status;
}

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_CODEC_SELECT_WRAPPER_H