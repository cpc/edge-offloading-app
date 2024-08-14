#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "fcntl.h"
#include "sys/stat.h"
#include "unistd.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "rename_opencl.h"
#include <CL/cl.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "opencl_utils.hpp"
#include "platform.h"
#include "sharedUtils.h"

#include "codec_select_wrapper.h"
#include "jpegReader.h"
#include <Tracy.hpp>
#include <thread>

/* forward declaration */
cl_int test_pthread_bik();
void read_function(pocl_image_processor_context *ctx, codec_select_state_t *state, int *read_frame_count);

/* pcapp settings */
constexpr float FPS = 3.0f;
constexpr int NFRAMES = 100;
constexpr bool RETRY_ON_DISCONNECT = true;

/* runtime settings */
constexpr int do_segment = 1;
constexpr int quality = 80;
constexpr int rotation = 0;
constexpr compression_t compression_type = JPEG_COMPRESSION;
constexpr int config_flags =
    NO_COMPRESSION | ENABLE_PROFILING | compression_type;
// can be changed due to retry_on_disconnect
device_type_enum device_index = REMOTE_DEVICE;

/* setup settings */
constexpr int do_algorithm = 0;
constexpr int lock_codec = 0;
constexpr bool enable_eval = true;
constexpr int max_lanes = 1;
constexpr bool has_video_input = false;

/* Read input image, assumes app is in a build dir */
const char *inp_name = "../../android/app/src/main/assets/bus_640x480.jpg";

int main() {
    cl_int status;

    image_data_t image_data;
    JPEGReader jpegReader(inp_name);
    auto [inp_w, inp_h] = jpegReader.getDimensions();
    jpegReader.readImage(&image_data);

    /* Init */
    int fd =
        open("profile.csv", O_WRONLY | O_CREAT | O_APPEND, S_IWRITE | S_IREAD);

    if (fd == -1) {
        perror("Cannot open file");
        return 1;
    }
    // assuming build directory is pcapp/<cmake build dir>/pcapp
    std::vector<std::string> source_files = {
        "../../android/app/src/main/assets/kernels/copy.cl",
        "../../android/app/src/main/assets/kernels/compress_seg.cl"};
    auto codec_sources = read_files(source_files);
    assert(codec_sources[0].length() > 0 && "could not open files");

    event_array_t event_array;
    event_array_t eval_event_array;

    int frame_index = 0;
    int is_eval_frame = 0;
    int64_t last_image_timestamp = get_timestamp_ns();
    float iou;
    uint64_t size_bytes;
    host_ts_ns_t host_ts_ns;

    // used by the reader thread to figure out when to stop
    int read_frame_count = 0;

RETRY:

    pocl_image_processor_context *ctx = nullptr;

    char const *source_strings[] = {codec_sources.at(0).c_str(),
                                    codec_sources.at(1).c_str()};
    size_t const source_sizes[] = {codec_sources.at(0).size(),
                                   codec_sources.at(1).size()};

    status = create_pocl_image_processor_context(
        &ctx, max_lanes, inp_w, inp_h, config_flags, source_strings,
        source_sizes, fd, enable_eval, nullptr);
    assert(status == CL_SUCCESS);
    assert(ctx != nullptr);

    codec_select_state_t *state = nullptr;
    init_codec_select(config_flags, fd, do_algorithm, lock_codec,
                      has_video_input, &state);

    // create read thread

    std::thread thread(read_function, ctx, state, &read_frame_count);

    /* Process */
    while (frame_index < NFRAMES) {
        int64_t image_timestamp = get_timestamp_ns();
        auto since_last_frame_sec =
            (float)(image_timestamp - last_image_timestamp) / 1e9f;

        if (since_last_frame_sec < (1.0f / FPS)) {
            continue;
        }

        FrameMark;
        // set a long timeout so that the loop is at minimum
        // 30 fps (see above code) and maximum as long as it takes
        // for another frame
        if (dequeue_spot(ctx, 20000, device_index) != 0) {
            continue;
        }

        last_image_timestamp = image_timestamp;
        image_data.image_timestamp = image_timestamp;

        codec_select_submit_image(state, ctx, device_index, do_segment,
                                  compression_type, quality, rotation,
                                  do_algorithm, &image_data);

        frame_index += 1;
        printf("submitted image : %d\n", frame_index);

        //        assert(status == CL_SUCCESS);
        log_if_cl_err(status, "main.c could not submit image");
        if (CL_SUCCESS != status) {
            break;
        }
    }

    // join the thread
    thread.join();

    status = destroy_pocl_image_processor_context(&ctx);

    if (true == RETRY_ON_DISCONNECT && (frame_index < NFRAMES)) {
        device_index = LOCAL_DEVICE;
        goto RETRY;
    }

    assert(status == CL_SUCCESS);

    close(fd);

    return status;
}

/**
 * The function that receives images continuously
 * @param ctx image processor context
 */
void read_function(pocl_image_processor_context *ctx, codec_select_state_t *state, int *read_frame_count) {

    int status;
    std::vector<int32_t> detections(DET_COUNT);
    std::vector<uint8_t> segmentations(SEG_OUT_COUNT);
    int64_t metadata_array[2];

    bool error_state = false;

    while (*read_frame_count < NFRAMES) {

        if (wait_image_available(ctx, 100) != 0) {
            if (!error_state) {
                continue;
            } else {
                break;
            }
        }

        detections[0] = 0;

        status = codec_select_receive_image(state, ctx, detections.data(),
                                            segmentations.data(), metadata_array);

        *read_frame_count += 1;

        log_if_cl_err(status, "main.c could not enqueue image");
        printf("==== Frame %d, no. detections: %d ====\n", *read_frame_count,
               detections[0]);

        if (CL_SUCCESS != status) {
            error_state = true;
            continue;
        }

        cv::Mat seg_out(MASK_SZ2, MASK_SZ1, CV_8UC4, segmentations.data());
        cv::cvtColor(seg_out, seg_out, cv::COLOR_RGBA2RGB);
        cv::imwrite("seg_out.png", seg_out);
    }
}

cl_int test_pthread_bik() {
    ZoneScoped;
    const int platform_id = 0;
    const int device_id = 0;

    /* Find platforms */
    cl_uint nplatforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &nplatforms);
    log_if_cl_err(status, "clGetPlatformIDs query");
    throw_if_zero(nplatforms, "No platforms found");

    /* Fill in platforms */
    std::vector<cl_platform_id> platforms;
    platforms.resize(nplatforms);
    status = clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
    log_if_cl_err(status, "clGetPlatformIDs fill in");

    /* Print out some basic information about each platform */
    status = print_platforms_info(platforms.data(), platforms.size());
    log_if_cl_err(status, "Getting platform info");

    /* Find devices */
    cl_uint ndevices = 0;
    cl_device_type device_types = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
    status = clGetDeviceIDs(platforms[platform_id], device_types, 0, NULL,
                            &ndevices);
    log_if_cl_err(status, "clGetDeviceIDs query");
    throw_if_zero(ndevices, "No devices found");

    /* Fill in devices */
    std::vector<cl_device_id> devices;
    devices.resize(ndevices);
    status = clGetDeviceIDs(platforms[platform_id], device_types,
                            devices.size(), devices.data(), NULL);
    log_if_cl_err(status, "clGetDeviceIDs fill query");

    /* Print out some basic information about each device */
    status = print_devices_info(devices.data(), devices.size());
    log_if_cl_err(status, "Getting device info");

    const cl_context_properties cps[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[platform_id]),
        0};

    cl_context context =
        clCreateContext(cps, ndevices, devices.data(), NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    cl_program tmp_program = clCreateProgramWithBuiltInKernels(
        context, 1, &devices[device_id], "pocl.add.i8", &status);
    CHECK_AND_RETURN(status, "TMP creation of program failed");

    status = clBuildProgram(tmp_program, 1, &devices[device_id], nullptr,
                            nullptr, nullptr);
    CHECK_AND_RETURN(status, "TMP building of program failed");
    LOGI("Created and built program");

    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
                                                CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue tmp_command_queue = clCreateCommandQueueWithProperties(
        context, devices[device_id], properties, &status);
    CHECK_AND_RETURN(status, "TMP creating eval command queue failed");
    LOGI("Created CQ\n");

    cl_kernel tmp_kernel = clCreateKernel(tmp_program, "pocl.add.i8", &status);
    CHECK_AND_RETURN(status, "TMP creating eval kernel failed");
    LOGI("Created kernel\n");

    char A[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    char B[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    char C[8] = {0};

    cl_mem tmp_buf_a =
        clCreateBuffer(context, CL_MEM_READ_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer A");
    cl_mem tmp_buf_b =
        clCreateBuffer(context, CL_MEM_READ_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer B");
    cl_mem tmp_buf_c =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer C");
    LOGI("Created buffers\n");

    status = clSetKernelArg(tmp_kernel, 0, sizeof(cl_mem), &tmp_buf_a);
    status |= clSetKernelArg(tmp_kernel, 1, sizeof(cl_mem), &tmp_buf_b);
    status |= clSetKernelArg(tmp_kernel, 2, sizeof(cl_mem), &tmp_buf_c);
    CHECK_AND_RETURN(status, "TMP could not assign kernel args");
    LOGI("Set kernel args\n");

    for (int i = 0; i < ndevices; ++i) {
        status = clReleaseDevice(devices[i]);
        CHECK_AND_RETURN(status, "TMP releasing device");
    }

    status = clEnqueueWriteBuffer(tmp_command_queue, tmp_buf_a, CL_TRUE, 0, 8,
                                  A, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to write A buffer");
    status = clEnqueueWriteBuffer(tmp_command_queue, tmp_buf_b, CL_TRUE, 0, 8,
                                  B, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to write B buffer");
    LOGI("Wrote buffers\n");

    const size_t global_size = 1;
    const size_t local_size = 1;
    status = clEnqueueNDRangeKernel(tmp_command_queue, tmp_kernel, 1, NULL,
                                    &global_size, &local_size, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to enqueue ND range kernel");
    LOGI("Enqueued kernel\n");

    status = clEnqueueReadBuffer(tmp_command_queue, tmp_buf_c, CL_TRUE, 0, 8, C,
                                 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to read C buffer");
    LOGI("Read buffers\n");

    for (int i = 0; i < 8; ++i) {
        LOGI("C[%d]: %d\n", i, C[i]);
        if (C[i] != 3) {
            LOGE("C[%d]: Wrong result!\n", i);
        }
    }

    status = clReleaseCommandQueue(tmp_command_queue);
    CHECK_AND_RETURN(status, "TMP failed to release CQ");
    status = clReleaseMemObject(tmp_buf_a);
    CHECK_AND_RETURN(status, "TMP failed to release buf a");
    status = clReleaseMemObject(tmp_buf_b);
    CHECK_AND_RETURN(status, "TMP failed to release buf b");
    status = clReleaseMemObject(tmp_buf_c);
    CHECK_AND_RETURN(status, "TMP failed to release buf c");
    status = clReleaseContext(context);
    CHECK_AND_RETURN(status, "TMP failed to release context");
    status = clReleaseKernel(tmp_kernel);
    CHECK_AND_RETURN(status, "TMP failed to release kernel");
    status = clReleaseProgram(tmp_program);
    CHECK_AND_RETURN(status, "TMP failed to release program");

    LOGI(">>>>> TMP SUCCESS <<<<<<\n");
    return CL_SUCCESS;
}