/**
 * Code related to asynchronous quality evaluation pipeline.
 */

#ifndef POCL_AISA_DEMO_EVAL_H
#define POCL_AISA_DEMO_EVAL_H

#include "poclImageProcessorV2.h"
#include "codec_select.h"

#define EVAL_VERBOSITY 0
#define EVAL_INTERVAL_SEC 2  // how often to run the quality eval in seconds

#ifdef __cplusplus
extern "C" {
#endif

cl_int check_eval(eval_pipeline_context_t *eval_ctx, codec_select_state_t *state,
                  const codec_config_t codec_config, bool *is_eval_frame);

cl_int run_eval(eval_pipeline_context_t *eval_ctx, codec_select_state_t *state,
                const codec_config_t codec_config, int frame_index, const image_data_t image_data);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_EVAL_H
