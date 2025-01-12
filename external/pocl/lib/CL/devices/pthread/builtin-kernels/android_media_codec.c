//
// Created by rabijl on 22.9.2023.
//

#include <string.h>
#include "android_media_codec.h"
#include <stdlib.h>
//#include "pocl_debug.h"
#include "time.h"
#include "media/NdkMediaFormat.h"

// define this to start printing stream to a file
//#define DEBUG

#define MIN(a, b)    \
        b < a ? b: a

#define US_IN_S 1000000

// https://developer.android.com/reference/android/media/MediaCodecInfo.CodecCapabilities#COLOR_FormatYUV422Flexible
#define COLOR_FormatYUV420Semiplanar 21
#define BITRATE_MODE_CBR 2
#define BITRATE_MODE_CQ 0

#define CHECK_AND_LOG_MEDIA(ret, msg) \
    if( AMEDIA_OK != ret) {           \
        POCL_MSG_ERR("Mediacodec error : %s \n", msg); \
    }                                 \

// TODO: store these states in either the program or somewhere else
static media_codec_state_t MEDIA_CODEC_STATE = {NULL, 0, 0, NULL, NULL};

static media_codec_state_t C2_ANDROID_HEVC_MEDIA_CODEC_STATE = {NULL, 0, 0, NULL, NULL};

void _pocl_kernel_pocl_encode_hevc_yuv420nv21_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {
    void **arguments = *(void ***) (args);
    void **arguments2 = (void **) (args);

    int nargs = 0;
    const uint8_t *input_buf = (const uint8_t *) (arguments[nargs++]);
    uint64_t input_size = *(uint64_t *) (arguments2[nargs++]);
    uint8_t *output_buf = (uint8_t *) (arguments[nargs++]);
    uint64_t output_buf_size = *(uint64_t *) (arguments2[nargs++]);
    uint64_t *data_written = (uint64_t *) (arguments[nargs++]);

    encode_image(&MEDIA_CODEC_STATE, input_buf, input_size, output_buf,
                 output_buf_size, data_written);
}

void init_mediacodec_encoder(cl_program program, cl_uint device_i) {
    create_media_codec(&MEDIA_CODEC_STATE);
}

void destroy_mediacodec_encoder(cl_device_id device, cl_program program,
                                unsigned dev_i) {
    destroy_codec(&MEDIA_CODEC_STATE);
}

void
_pocl_kernel_pocl_configure_hevc_yuv420nv21_workgroup(cl_uchar *args, cl_uchar *context, ulong group_x, ulong group_y,
                                                      ulong group_z) {

    void **arguments = (void **) (args);

    int nargs = 0;
    int32_t height = *(int32_t *) (arguments[nargs++]);
    int32_t width = *(int32_t *) (arguments[nargs++]);
    int32_t framerate = *(int32_t *) (arguments[nargs++]);
    int32_t i_frame_interval = *(int32_t *) (arguments[nargs++]);
    int32_t bitrate = *(int32_t *) (arguments[nargs++]);

    configure_codec(&MEDIA_CODEC_STATE, height, width, framerate, i_frame_interval, bitrate);

}

void init_c2_android_hevc_encoder(cl_program program, cl_uint device_i) {
    create_c2_android_hevc_encoder( &C2_ANDROID_HEVC_MEDIA_CODEC_STATE);
}

void destroy_c2_android_hevc_encoder(cl_device_id device, cl_program program,
                                     unsigned dev_i) {
    destroy_codec( &C2_ANDROID_HEVC_MEDIA_CODEC_STATE);
}

void
_pocl_kernel_pocl_encode_c2_android_hevc_yuv420nv21_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

    void **arguments = *(void ***) (args);
    void **arguments2 = (void **) (args);

    int nargs = 0;
    const uint8_t *input_buf = (const uint8_t *) (arguments[nargs++]);
    uint64_t input_size = *(uint64_t *) (arguments2[nargs++]);
    uint8_t *output_buf = (uint8_t *) (arguments[nargs++]);
    uint64_t output_buf_size = *(uint64_t *) (arguments2[nargs++]);
    uint64_t *data_written = (uint64_t *) (arguments[nargs++]);

    encode_image( &C2_ANDROID_HEVC_MEDIA_CODEC_STATE , input_buf, input_size, output_buf,
                 output_buf_size, data_written);

}

void
_pocl_kernel_pocl_configure_c2_android_hevc_yuv420nv21_workgroup(cl_uchar *args, cl_uchar *context, ulong group_x, ulong group_y,
                                                      ulong group_z) {

    void **arguments = (void **) (args);

    int nargs = 0;
    int32_t height = *(int32_t *) (arguments[nargs++]);
    int32_t width = *(int32_t *) (arguments[nargs++]);
    int32_t framerate = *(int32_t *) (arguments[nargs++]);
    int32_t i_frame_interval = *(int32_t *) (arguments[nargs++]);
    int32_t bitrate = *(int32_t *) (arguments[nargs++]);

    configure_codec(&C2_ANDROID_HEVC_MEDIA_CODEC_STATE, height, width, framerate, i_frame_interval, bitrate);

}

void
create_media_codec(media_codec_state_t *state) {

    media_status_t status;
    if (NULL != state->codec) {
        POCL_MSG_PRINT_INFO("android hevc encoder already exists, skipping \n");
        return;
    }

    state->format = NULL;

    state->codec = AMediaCodec_createEncoderByType("video/hevc");

    state->frame_clock = 0;

#ifdef DEBUG
    char file_name[64];
    time_t cur_time = time(NULL);
    sprintf(file_name, "/storage/self/primary/Download/%ld.bin", (long) cur_time);
    state->write_ptr = fopen(file_name, "wb");
#else
    state->write_ptr = NULL;
#endif

}

void create_c2_android_hevc_encoder(media_codec_state_t *state) {

    media_status_t status;
    if (NULL != state->codec) {
        POCL_MSG_PRINT_INFO("c2.android.hevc.encoder already exists, skipping \n");
        return;
    }

    state->format = NULL;

    state->codec = AMediaCodec_createCodecByName("c2.android.hevc.encoder");

    state->frame_clock = 0;

#ifdef DEBUG
    char file_name[64];
    time_t cur_time = time(NULL);
    sprintf(file_name, "/storage/self/primary/Download/%ld.bin", (long) cur_time);
    state->write_ptr = fopen(file_name, "wb");
#else
    state->write_ptr = NULL;
#endif

}

void
configure_codec(media_codec_state_t *state, int32_t width, int32_t height, int32_t framerate,
                int32_t i_frame_interval, int32_t bitrate) {

    media_status_t status;

    if (NULL != state->format) {
        status = AMediaFormat_delete(state->format);
        CHECK_AND_LOG_MEDIA(status, "error freeing format")
        state->format = NULL;
    }

    if (NULL == state->codec) {
        POCL_MSG_ERR("configure called before creation, creating codec");
        create_media_codec(state);
    }

    // todo: see if we can just use time
    // needed since the codec needs timestamps to compress images
    state->time_increment = US_IN_S / ((int64_t) framerate);

    state->format = AMediaFormat_new();

    // this is needed to prevent android from overriding the bitrate we set,
    // even though the hardware might not support constant bitrate rate.
    // The hardware will just fall back to what it supports (variable bitrate).
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_BITRATE_MODE, BITRATE_MODE_CBR);

    AMediaFormat_setString(state->format, AMEDIAFORMAT_KEY_MIME, "video/hevc"); // is supported
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_WIDTH, width); // 640 is supported
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_HEIGHT, height); // 480 is supported
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_COLOR_FORMAT,
                          COLOR_FormatYUV420Semiplanar);
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_FRAME_RATE, framerate);
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_CAPTURE_RATE, framerate);
    // an i frame every 2 seconds is recommended
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_I_FRAME_INTERVAL, i_frame_interval);
    // most likely the weird android quality "helper" bumps up the bitrate higher than this
    // but it is still compressed
    AMediaFormat_setInt32(state->format, AMEDIAFORMAT_KEY_BIT_RATE, bitrate);

    uint32_t flags = AMEDIACODEC_CONFIGURE_FLAG_ENCODE;

    // https://developer.android.com/reference/android/media/MediaCodec#asynchronous-processing-using-buffers
    status = AMediaCodec_stop(state->codec);
    CHECK_AND_LOG_MEDIA(status, "failed to stop codec")

    status = AMediaCodec_configure(state->codec, state->format, NULL, NULL, flags);
    CHECK_AND_LOG_MEDIA(status, "couldn't configure media codec")

    status = AMediaCodec_start(state->codec);
    CHECK_AND_LOG_MEDIA(status, "couldn't start media codec")

}

void
encode_image(media_codec_state_t *state, const uint8_t *input_buf, uint64_t input_size, uint8_t *output_buf,
             uint64_t output_buf_size, uint64_t *data_written) {

    if (NULL == state->format) {
        POCL_MSG_ERR(
                "codec was not set up before calling encoding image, configuring with defaults");
        configure_codec(state, 640, 480, 5, 2, 640 * 480);
    }

    // have a look at the type def of media_status_t for all the error types
    ssize_t index = -1;

    // the timeout is set with the assumption that it is the same as the java code
    // https://developer.android.com/reference/android/media/MediaCodec#dequeueInputBuffer(long)
    index = AMediaCodec_dequeueInputBuffer(state->codec, -1);

    if (index >= 0) {
        // if it is not, something went bad
        assert(index >= 0);

        size_t codec_inp_buf_size;
        uint8_t *codec_inp_buf = AMediaCodec_getInputBuffer(state->codec, index,
                                                            &codec_inp_buf_size);

        assert(codec_inp_buf_size >= input_size);

        size_t data_copied = MIN(input_size, codec_inp_buf_size);
        memcpy(codec_inp_buf, input_buf, data_copied);

        AMediaCodec_queueInputBuffer(state->codec, index, 0, data_copied, state->frame_clock, 0);
        state->frame_clock += state->time_increment;

    }

    AMediaCodecBufferInfo info;
    index = AMediaCodec_dequeueOutputBuffer(state->codec, &info, -1);

    if (index >= 0) {
        uint8_t *codec_out_buf = AMediaCodec_getOutputBuffer(state->codec, index,
                                                             NULL);

        assert(info.size > 0);
        assert(info.size < output_buf_size);

        memcpy(output_buf, codec_out_buf, info.size);
        *data_written = info.size;
        POCL_MSG_PRINT_INFO("encoded packet size: %d", info.size);

        AMediaCodec_releaseOutputBuffer(state->codec, index, 0);

#ifdef DEBUG
        if (NULL != state->write_ptr) {
            // write result to file
            fwrite(output_buf, 1, *data_written, state->write_ptr);
        }
#endif

        return;
    }

    if (AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED == index) {
        AMediaFormat *format = AMediaCodec_getOutputFormat(state->codec);
        POCL_MSG_ERR("format changed to : %s \n", AMediaFormat_toString(format));
        AMediaFormat_delete(format);
    } else if (AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED == index) {
        POCL_MSG_ERR("output buffer changed \n");
    } else if (AMEDIACODEC_INFO_TRY_AGAIN_LATER == index) {
        POCL_MSG_ERR("could not get an output buffer right now \n");
    } else {
        POCL_MSG_ERR("unknown error: %zd \n", index);
    }
    *data_written=0;
}

void destroy_codec(media_codec_state_t *state) {

    media_status_t status;

    if (NULL != state->format) {
        status = AMediaFormat_delete(state->format);
        CHECK_AND_LOG_MEDIA(status, "could not delete media format")
        state->format = NULL;
    }

    if (NULL != state->codec) {
        status = AMediaCodec_stop(state->codec);
        POCL_MSG_PRINT_INFO("stop status: %d \n", status);

        status = AMediaCodec_delete(state->codec);
        POCL_MSG_PRINT_INFO("delete status: %d \n", status);

        state->codec = NULL;
    }

#ifdef DEBUG
    if (NULL != state->write_ptr) {
        fclose(state->write_ptr);
        state->write_ptr = NULL;
    }
#endif

}
