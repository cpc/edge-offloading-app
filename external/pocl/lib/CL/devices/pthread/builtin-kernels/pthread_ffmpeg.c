//
// Created by rabijl on 18.10.2023.
//

#include <assert.h>
#include "pthread_ffmpeg.h"
#include "pocl_debug.h"

#define CODEC_NAME "hevc_cuvid"
//#define DEBUG

// TODO: store this either in program or something else like kernel
static ffmpeg_state_t FFMPEG_STATE = {0, NULL, NULL, NULL, NULL, NULL, PTHREAD_MUTEX_INITIALIZER};

void _pocl_kernel_pocl_decode_hevc_yuv420nv21_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z) {
  void **arguments = *(void ***)(args);
  void **arguments2 = (void **)(args);

  int nargs = 0;
  const uint8_t *input = (const uint8_t *)(arguments[nargs++]);
  uint64_t *input_size = (uint64_t *)(arguments[nargs++]);
  uint8_t *output = (uint8_t *)(arguments[nargs++]);
  uint64_t output_size = *(uint64_t *)(arguments2[nargs++]);

  decode_hevc(&FFMPEG_STATE, input, input_size, output, output_size);
}

void destroy_ffmpeg_decoder(cl_device_id device, cl_program program,
                            unsigned dev_i) {
  destroy_ffmpeg(&FFMPEG_STATE);
}

static void write_frame (AVFrame *frame, char *file_prefix, int frame_number)
{
  if (NULL == file_prefix)
    {
      return;
    }

  char out_file_name[1024];
  snprintf (out_file_name, sizeof (out_file_name), "%s-%d.pgm", file_prefix, frame_number);
  printf ("file name: %s \n", out_file_name);
  FILE *file = fopen (out_file_name, "wb");
    if (NULL == file) {
        POCL_MSG_ERR("could not open output image file: %s\n", out_file_name);
        return;
    }

    // add the magic packets for a portable gray map file (pgm)
    fprintf(file, "P5\n%d %d\n%d\n", frame->width, frame->height, 255);

    // just the Y part
    for (int i = 0; i < frame->height; i++) {
        fwrite(frame->data[0] + frame->linesize[0] * i, 1, frame->width, file);
    }

    fclose(file);
}

static void copy_frame_to_array(AVFrame *frame, uint8_t *output, uint64_t output_size) {

    // wipe buffer
    if (NULL == frame) {
        memset(output, '0', output_size);
        return;
    }

    // make sure that the frame decoded to yuv420
    assert(AV_PIX_FMT_YUV420P == frame->format);
    // make sure it fits in the output buffer
    assert(output_size >= (frame->width * frame->height) * 3 / 2);

    int total_pixels = frame->height * frame->width;
    // Y
    memcpy(output, frame->data[0], total_pixels);
    // U
    memcpy(output + total_pixels, frame->data[1], total_pixels / 4);
    // V
    memcpy(output + total_pixels + total_pixels / 4, frame->data[2], total_pixels / 4);

    return;
}

static int decode(ffmpeg_state_t *state, uint8_t *output, uint64_t output_size) {
    AVCodecContext *dec_ctx = state->decoder_context;
    AVFrame *frame = state->frame;
    AVPacket *packet = state->packet;

    int ret;

    ret = avcodec_send_packet(dec_ctx, packet);
    if (AVERROR_EOF == ret) {
        POCL_MSG_ERR("decoder has been flushed\n");
        return -1;
    } else if (AVERROR(EAGAIN) == ret) {
        POCL_MSG_PRINT_INFO("avcodec try send packet again\n");
    } else if (ret < 0) {
        POCL_MSG_ERR("Error sending a packet for decoding, err: %d\n", ret);
        return -1;
    }
#ifdef DEBUG
    POCL_MSG_PRINT_INFO("sent packet to decoder\n");
#endif

    ret = avcodec_receive_frame(dec_ctx, frame);
    // these errors are ok
    if (ret == AVERROR(EAGAIN)) {
#ifdef DEBUG
        POCL_MSG_WARN("avcodec try again\n");
#endif
        // if there is no frame, fill the buffer with zeros
        copy_frame_to_array(NULL, output, output_size);
        return 0;
    } else if (ret == AVERROR_EOF) {
#ifdef DEBUG
        POCL_MSG_WARN("avcodec_receive_frame eof\n");
#endif
        return 0;
    }
    // these are not
    else if (ret < 0) {
        POCL_MSG_WARN("Error during decoding\n");
        return -1;
    }

#ifdef DEBUG
    POCL_MSG_WARN("frame pixel format: %d\n", frame->format);
    write_frame(frame, state->out_file_prefix, dec_ctx->frame_number);
#endif

    // copy frame to output
    copy_frame_to_array(frame, output, output_size);

    // make sure that there is nothing left in the decoder
    ret = avcodec_receive_frame(dec_ctx, frame);
    if (ret != AVERROR(EAGAIN)) {
        POCL_MSG_WARN("there was a second frame in the packet "
                      "that was not copied to the output buffer\n");
    }

}

int
init_ffmpeg(ffmpeg_state_t *state, const uint8_t *input,
                 const uint64_t *input_size)
{
    int status = 0;

    // set everything to NULL incase something goes wrong during init,
    // and we have to free the state
    state->opened = 0;
    state->parser_context = NULL;
    state->decoder_context = NULL;
    state->packet = NULL;
    state->frame = NULL;

    state->packet = av_packet_alloc();
    if (NULL == state->packet) {
        POCL_MSG_ERR("could not alloc av_packet\n");
        return -1;
    }

    // it turns out that the hardware decoder returns images in AV_PIX_FMT_NV12 format
    // while the software decoder does AV_PIX_FMT_YUV420P
//  const AVCodec *codec = avcodec_find_decoder_by_name(CODEC_NAME);
    const AVCodec *codec = NULL;
    if (NULL == codec) {
        POCL_MSG_WARN("could not find codec: %s \n trying generic hevc decoder.\n", CODEC_NAME);
        codec = avcodec_find_decoder(AV_CODEC_ID_HEVC);
    }

    if (NULL == codec) {
        POCL_MSG_ERR("could not find HEVC decoder\n");
        return -1;
    }

    state->parser_context = av_parser_init (codec->id);
  if (NULL == state->parser_context)
    {
      POCL_MSG_ERR("could not init av_parser\n");
      return -1;
    }

  state->decoder_context = avcodec_alloc_context3 (codec);
  if (NULL == state->decoder_context)
    {
      POCL_MSG_ERR("could not alloc av_decoder context\n");
      return -1;
    }

  status = avcodec_open2 (state->decoder_context, codec, NULL);
  if (status < 0)
    {
      POCL_MSG_ERR("could not open av_decoder context\n");
      return -1;
    }

  state->frame = av_frame_alloc ();
  if (NULL == state->frame)
    {
      POCL_MSG_ERR("could not alloc av_frame\n");
      return -1;
    }

  state->out_file_prefix = "frame";
  state->opened = 1;

  return 0;
}

void destroy_ffmpeg(ffmpeg_state_t *state)
{

  // todo: possibly flush the contents

  state->opened = 0;

  if (NULL != state->parser_context)
    {
      av_parser_close (state->parser_context);
      state->parser_context = NULL;
    }

  if (NULL != state->decoder_context)
    {
      avcodec_free_context (&(state->decoder_context));
      state->decoder_context = NULL;
    }

  if (NULL != state->packet)
    {
      av_packet_free (&(state->packet));
      state->packet = NULL;
    }

  if (NULL != state->frame)
    {
      av_frame_free (&(state->frame));
      state->frame = NULL;
    }

    pthread_mutex_destroy(&(state->execution_lock));

}

#ifdef DEBUG
size_t total_bytes_read = 0;
#endif

int
decode_hevc(ffmpeg_state_t *state,
            const uint8_t *input,
            const uint64_t *input_size,
            uint8_t *output,
            const uint64_t output_size) {

    int status = 0;

    // if things are not set up, do that now
    if (1 != state->opened) {
        POCL_MSG_ERR("codec is not initialized! \n");
        return -1;
    }

    pthread_mutex_lock(&(state->execution_lock));

    int ret;

    size_t data_size_remaining = *input_size;
    // create a pointer to the buffer that we keep incrementing
    // every iteration of the while loop
    const uint8_t *data = input;

    while (data_size_remaining > 0) {

        // start parsing the data
        ret = av_parser_parse2(state->parser_context,
                               state->decoder_context,
                               &(state->packet->data),
                               &(state->packet->size),
                               data,
                               data_size_remaining,
                               AV_NOPTS_VALUE,
                               AV_NOPTS_VALUE,
                               0);

        if (ret < 0) {
            POCL_MSG_ERR("could not parse packet \n");
            return -1;
        }

        data += ret;
        data_size_remaining -= ret;

#ifdef DEBUG
        total_bytes_read += ret;
#endif

        // if not enough data has been read for a packet to be parsed,
        // continue parsing
        if (0 == state->packet->size) {
            continue;
        }

#ifdef  DEBUG
        POCL_MSG_PRINT_INFO("packet size: %d, pic type: %c, pic num: %d \n", state->packet->size,
                            av_get_picture_type_char(state->parser_context->pict_type),
                            state->parser_context->output_picture_number);
#endif

        decode(state, output, output_size);
    }

#ifdef DEBUG
    POCL_MSG_PRINT_INFO("total bytes read so far %lu\n", total_bytes_read);
#endif

    pthread_mutex_unlock(&(state->execution_lock));

    return 0;
}

POCL_EXPORT
void kick_ffmpeg_awake() {
    POCL_MSG_PRINT_INFO("kicking ffmpeg awake\n");
    init_ffmpeg(&FFMPEG_STATE, 0, 0);
}