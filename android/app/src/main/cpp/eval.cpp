#include "eval.h"
#include "dnn_stage.hpp"
#include "sharedUtils.h"

#ifdef __cplusplus
extern "C" {
#endif

static bool is_eval_running(const eval_pipeline_context_t *eval_ctx) {
    return eval_ctx->is_eval_running;
}

static bool
should_run_eval(const eval_pipeline_context_t *eval_ctx, const bool eval_every_frame,
                const codec_config_t *codec_config) {
    if (LOCAL_DEVICE == codec_config->device_type) {
        return false;
    } else if (eval_every_frame) {
        return true;
    } else {
        timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        return (compare_timespec(&ts, &(eval_ctx->next_eval_ts)) > 0 && !eval_ctx->is_eval_running);
    }
}

static cl_int enqueue_eval_dnn_iou(eval_pipeline_context_t *eval_ctx, const codec_config_t *config,
                                   event_array_t *event_array, float *iou) {

    const dnn_context_t *const eval_dnn_ctx = eval_ctx->eval_pipeline->dnn_context;

    cl_int status;
    status = clSetKernelArg(eval_dnn_ctx->eval_kernel, 0, sizeof(cl_mem),
                            &(eval_dnn_ctx->detect_buf));
    status |= clSetKernelArg(eval_dnn_ctx->eval_kernel, 1, sizeof(cl_mem),
                             &(eval_dnn_ctx->postprocess_buf));
    status |= clSetKernelArg(eval_dnn_ctx->eval_kernel, 2, sizeof(cl_mem),
                             &(eval_ctx->tmp_buf_ctx.det));
    status |= clSetKernelArg(eval_dnn_ctx->eval_kernel, 3, sizeof(cl_mem),
                             &(eval_ctx->tmp_buf_ctx.seg_post));
    status |= clSetKernelArg(eval_dnn_ctx->eval_kernel, 4, sizeof(cl_int), &(config->do_segment));
//    status |= clSetKernelArg(dnn_context->eval_kernel, 5, sizeof(cl_mem),
//                             &(dnn_context->eval_buf));
    CHECK_AND_RETURN(status, "could not assign eval kernel params");

    // figure out on which queue to run the dnn
    cl_command_queue queue;
    PICK_QUEUE(queue, eval_dnn_ctx, config);

    // TODO: add check to see if queue is present in eval_dnn_ctx as well
    // and give a warning that unexpected buffer transfers might happen
    // if this is not the case

    cl_event eval_iou_event, iou_read_event;

    {
        int wait_list_size = 3;
        cl_event wait_list[3] = {eval_ctx->tmp_buf_ctx.copy_event_det,
                                 eval_ctx->tmp_buf_ctx.copy_event_seg_post,
                                 eval_ctx->dnn_out_event};
//        TracyCLZone(dnn_ctx->remote_tracy_ctx, "eval IoU");
        status = clEnqueueNDRangeKernel(queue, eval_dnn_ctx->eval_kernel, eval_dnn_ctx->work_dim,
                                        NULL, eval_dnn_ctx->global_size, eval_dnn_ctx->local_size,
                                        wait_list_size, wait_list, &eval_iou_event);
        CHECK_AND_RETURN(status, "failed to enqueue ND range eval kernel");
        append_to_event_array(event_array, eval_iou_event, VAR_NAME(eval_iou_event));
//        TracyCLZoneSetEvent(eval_iou_event);
    }

    status = clEnqueueReadBuffer(queue, eval_dnn_ctx->eval_buf, CL_FALSE, 0, 1 * sizeof(cl_float),
                                 iou, 1, &eval_iou_event, &iou_read_event);
    CHECK_AND_RETURN(status, "failed to read eval iou result buffer");
    append_to_event_array(event_array, iou_read_event, VAR_NAME(iou_read_event));
    eval_ctx->iou_read_event = iou_read_event;

    return CL_SUCCESS;
}

static cl_int
submit_eval_frame(eval_pipeline_context_t *eval_ctx, const codec_config_t codec_config,
                  const image_data_t image_data) {
    cl_int status;

    codec_config_t eval_config = {NO_COMPRESSION, codec_config.device_type, codec_config.rotation,
                                  codec_config.do_segment, {.jpeg = {.quality = 0}},
                                  LOCAL_CODEC_ID};

    dnn_results eval_results;
    eval_results.event_list_size = 1;
    status = submit_image_to_pipeline((eval_ctx->eval_pipeline), eval_config, false, image_data,
                                      NULL, &eval_results, NULL);
    CHECK_AND_RETURN(status, "could not submit eval image");
    eval_ctx->dnn_out_event = eval_results.event_list[0];

    status = enqueue_eval_dnn_iou(eval_ctx, &codec_config, eval_ctx->eval_pipeline->event_array,
                                  &eval_ctx->iou);
    CHECK_AND_RETURN(status, "could not enqueue eval kernel");

    return CL_SUCCESS;
}

cl_int check_eval(eval_pipeline_context_t *eval_ctx, codec_select_state_t *state,
                  const codec_config_t codec_config, const bool eval_every_frame,
                  bool *is_eval_frame) {
    ZoneScoped;
    cl_int status;
    *is_eval_frame = false;

    if (eval_ctx == NULL) {
        // local only
        return CL_SUCCESS;
    }

    if (is_eval_running(eval_ctx)) {
        ZoneScopedN("eval poll");

        // eval is running, check if it's ready
        cl_int eval_status;
        status = clGetEventInfo(eval_ctx->iou_read_event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                                sizeof(cl_int), &eval_status, NULL);
        CHECK_AND_RETURN(status, "could not get eval event info");

        if (eval_status == CL_COMPLETE || eval_every_frame) {
            ZoneScopedN("eval poll complete");

            // make sure we're synchronized
            {
                ZoneScopedN("eval poll wait");
                clWaitForEvents(1, &eval_ctx->iou_read_event);
            }

            // eval finished
            eval_ctx->is_eval_running = false;
            signal_eval_finish(state, eval_ctx->iou);

            TracyCLCollect(eval_ctx->eval_pipeline->dnn_context->local_tracy_ctx);
            TracyCLCollect(eval_ctx->eval_pipeline->dnn_context->remote_tracy_ctx);

            release_events(eval_ctx->tmp_buf_ctx.event_array);
            reset_event_array(eval_ctx->tmp_buf_ctx.event_array);
            release_events(eval_ctx->event_array);
            reset_event_array(eval_ctx->event_array);

            if (EVAL_VERBOSITY >= 1) {
                LOGI("=== EVAL IOU: finished, iou: %5.3f\n", eval_ctx->iou);
            }
        } else {
            // eval not finished
            if (EVAL_VERBOSITY >= 1) {
                LOGI("=== EVAL IOU: still running... event status %d\n", eval_status);
            }
        }
    }

    if (should_run_eval(eval_ctx, eval_every_frame, &codec_config)) {
        if (EVAL_VERBOSITY >= 1) {
            LOGI("=== EVAL IOU: signaling to start\n");
        }

        *is_eval_frame = true;
    }

    return CL_SUCCESS;
}

cl_int run_eval(eval_pipeline_context_t *eval_ctx, codec_select_state_t *state,
                const codec_config_t codec_config, int frame_index, const image_data_t image_data) {
    cl_int status;

    if (EVAL_VERBOSITY >= 1) {
        LOGI("=== EVAL IOU: start\n");
    }

    status = submit_eval_frame(eval_ctx, codec_config, image_data);
    CHECK_AND_RETURN(status, "could not submit eval frame");

    // set this to true only after we know that submitting went fine
    eval_ctx->is_eval_running = true;
    int codec_id = get_codec_id(state);
    signal_eval_start(state, frame_index, codec_id);

    // increment the time for the next eval
    clock_gettime(CLOCK_MONOTONIC, &eval_ctx->next_eval_ts);
    eval_ctx->next_eval_ts.tv_sec += EVAL_INTERVAL_SEC;

    return CL_SUCCESS;
}

#ifdef __cplusplus
}
#endif