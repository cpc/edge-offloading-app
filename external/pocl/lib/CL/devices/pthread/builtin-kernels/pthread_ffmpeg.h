//
// Created by rabijl on 18.10.2023.
//

#ifndef _BASIC_FFMPEG_H_
#define _BASIC_FFMPEG_H_


#ifdef __cplusplus
extern "C" {
#endif

#include <CL/cl.h>
#include <pocl_types.h>
#include <pocl_cl.h>
//#include <libavformat/avformat.h>
//#include <libavformat/avio.h>
#include <libavcodec/avcodec.h>
#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

/**
* todo: add more needed user data for callback
*/
typedef struct {
  uint8_t *ptr;
  size_t capacity;
} callback_data_t;

typedef struct {
  /**
   * a variable used to indicate that ffmpeg is opened and ready
   */
  int opened;

  /**
   * the context containing the parser that parses the input
   */
  AVCodecParserContext *parser_context;

  /**
   * the context used to create the decoder
   */
  AVCodecContext *decoder_context;

  /**
   * a packet of image data to be decoded
   */
  AVPacket *packet;

    /**
     * the decoded frame
     */
    AVFrame *frame;

    /**
     * used for debugging
     */
    char *out_file_prefix;

    /**
     * don't decode two things at the same time
     */
    pthread_mutex_t execution_lock;

} ffmpeg_state_t;

POCL_EXPORT
void _pocl_kernel_pocl_decode_hevc_yuv420nv21_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void destroy_ffmpeg_decoder(cl_device_id device, cl_program program,
                            unsigned dev_i);

int
init_ffmpeg(ffmpeg_state_t *state, const uint8_t *input,
            const uint64_t *input_size);

void destroy_ffmpeg(ffmpeg_state_t *state);

int
decode_hevc(ffmpeg_state_t *state,
            const uint8_t *input,
            const uint64_t *input_size,
            uint8_t *output,
            const uint64_t output_size);

POCL_EXPORT
void kick_ffmpeg_awake();

#ifdef __cplusplus
}
#endif

#endif //_BASIC_FFMPEG_H_
