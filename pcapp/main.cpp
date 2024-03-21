#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "fcntl.h"
#include "sys/stat.h"
#include "unistd.h"

#include "rename_opencl.h"

#include <CL/cl.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "opencl_utils.hpp"
#include "platform.h"
#include "poclImageProcessor.h"
#include "sharedUtils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <Tracy.hpp>
#include <poclImageProcessorV2.h>
#include <thread>

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

constexpr int DETECTION_COUNT = 1 + MAX_DETECTIONS * 6;
constexpr int SEGMENTATION_COUNT = MAX_DETECTIONS * MASK_W * MASK_H;
constexpr int SEG_POSTPROCESS_COUNT = 4 * MASK_W * MASK_H; // RGBA image

constexpr float FPS = 30.0f;
constexpr int NFRAMES = 100;
// constexpr int NFRAMES = 10;

cl_int test_pthread_bik() {
    ZoneScoped;
    const int platform_id = 0;
    const int device_id = 0;

    /* Find platforms */
    cl_uint nplatforms = 0;
    cl_int status = clGetPlatformIDs(0, NULL, &nplatforms);
    throw_if_cl_err(status, "clGetPlatformIDs query");
    throw_if_zero(nplatforms, "No platforms found");

    /* Fill in platforms */
    std::vector<cl_platform_id> platforms;
    platforms.resize(nplatforms);
    status = clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
    throw_if_cl_err(status, "clGetPlatformIDs fill in");

    /* Print out some basic information about each platform */
    status = print_platforms_info(platforms.data(), platforms.size());
    throw_if_cl_err(status, "Getting platform info");

    /* Find devices */
    cl_uint ndevices = 0;
    cl_device_type device_types = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
    status = clGetDeviceIDs(platforms[platform_id], device_types, 0, NULL,
                            &ndevices);
    throw_if_cl_err(status, "clGetDeviceIDs query");
    throw_if_zero(ndevices, "No devices found");

    /* Fill in devices */
    std::vector<cl_device_id> devices;
    devices.resize(ndevices);
    status = clGetDeviceIDs(platforms[platform_id], device_types,
                            devices.size(), devices.data(), NULL);
    throw_if_cl_err(status, "clGetDeviceIDs fill query");

    /* Print out some basic information about each device */
    status = print_devices_info(devices.data(), devices.size());
    throw_if_cl_err(status, "Getting device info");

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

    cl_command_queue_properties cq_properties = CL_QUEUE_PROFILING_ENABLE;
    cl_command_queue tmp_command_queue = clCreateCommandQueue(
        context, devices[device_id], cq_properties, &status);
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

/**
 * The function that receives images continuously
 * @param ctx image processor context
 */
void read_function(pocl_image_processor_context *ctx) {

    int frame_index = 0;
    int segmentation = 0;
    eval_metadata_t eval_metadata;
    int status;
    std::vector<int32_t> detections(DETECTION_COUNT);
    std::vector<uint8_t> segmentations(SEG_POSTPROCESS_COUNT);

    while (frame_index < NFRAMES) {

        if (wait_image_available(ctx, 1000) != 0) {
            continue;
        }
        status = receive_image(ctx, detections.data(), segmentations.data(),
                               &eval_metadata, &segmentation);

        throw_if_cl_err(status, "could not enqueue image");
        printf("==== Frame %d, no. detections: %d ====\n", frame_index,
               detections[0]);

        cv::Mat seg_out(MASK_H, MASK_W, CV_8UC4, segmentations.data());
        cv::cvtColor(seg_out, seg_out, cv::COLOR_RGBA2RGB);
        cv::imwrite("seg_out.png", seg_out);

        frame_index += 1;
    }
}

int main() {
    cl_int status;

    // Sanity check run:
    // status = test_pthread_bik();
    // CHECK_AND_RETURN(status, "Error running vector addition");

    int platform_id = 0;
    // int enc_device_id = 0; // on my machine pthread device
    // int dec_device_id = 0; // on my machine pthread device
    // int dnn_device_id = 0; // on my machine basic device

    /* Find platforms */
    cl_uint nplatforms = 0;
    status = clGetPlatformIDs(0, NULL, &nplatforms);
    throw_if_cl_err(status, "clGetPlatformIDs query");
    throw_if_zero(nplatforms, "No platforms found");

    /* Fill in platforms */
    std::vector<cl_platform_id> platforms;
    platforms.resize(nplatforms);
    status = clGetPlatformIDs(platforms.size(), platforms.data(), NULL);
    throw_if_cl_err(status, "clGetPlatformIDs fill in");

    /* Print out some basic information about each platform */
    status = print_platforms_info(platforms.data(), platforms.size());
    throw_if_cl_err(status, "Getting platform info");

    /* Find devices */
    cl_uint ndevices = 0;
    cl_device_type device_types = CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU;
    status = clGetDeviceIDs(platforms[platform_id], device_types, 0, NULL,
                            &ndevices);
    throw_if_cl_err(status, "clGetDeviceIDs query");
    throw_if_zero(ndevices, "No devices found");

    /* Fill in devices */
    std::vector<cl_device_id> devices;
    devices.resize(ndevices);
    status = clGetDeviceIDs(platforms[platform_id], device_types,
                            devices.size(), devices.data(), NULL);
    throw_if_cl_err(status, "clGetDeviceIDs fill query");

    /* Print out some basic information about each device */
    status = print_devices_info(devices.data(), devices.size());
    throw_if_cl_err(status, "Getting device info");

    /* Settings */
    constexpr compression_t compression_type = YUV_COMPRESSION;
    //    constexpr compression_t compression_type = NO_COMPRESSION;
    //    constexpr compression_t compression_type = JPEG_COMPRESSION;
    constexpr int config_flags = ENABLE_PROFILING | compression_type;

    constexpr devic_type_enum device_index = REMOTE_DEVICE;
    //    constexpr devic_type_enum device_index = LOCAL_DEVICE;
    constexpr int do_segment = 1;
    constexpr int quality = 80;
    constexpr int rotation = 0;

    /* Read input image, assumes app is in a build dir */
    std::string inp_name = "../../android/app/src/main/assets/bus_640x480.jpg";

    int inp_w, inp_h, nch;
    uint8_t *inp_pixels = stbi_load(inp_name.data(), &inp_w, &inp_h, &nch, 3);
    throw_if_nullptr(inp_pixels, "Error opening input file");
    printf("OpenCL: Opened file '%s', %dx%d\n", inp_name.c_str(), inp_w, inp_h);

    // Convert inp image to YUV420 (U/V planes separate)
    cv::Mat inp_img(inp_h, inp_w, CV_8UC3, (void *)(inp_pixels));
    cv::cvtColor(inp_img, inp_img, cv::COLOR_RGB2YUV_YV12);
    free(inp_pixels);

    // Convert separate U/V planes into interleaved U/V planes
    uint8_t *inp_yuv420nv21 = (uint8_t *)(malloc(inp_w * inp_h * 3 / 2));
    memcpy(inp_yuv420nv21, inp_img.data, inp_w * inp_h);

    for (int i = 0; i < inp_w * inp_h / 4; i += 1) {
        inp_yuv420nv21[inp_w * inp_h + 2 * i] =
            inp_img.data[inp_w * inp_h + inp_w * inp_h / 4 + i];
        inp_yuv420nv21[inp_w * inp_h + 2 * i + 1] =
            inp_img.data[inp_w * inp_h + i];
    }

    const int inp_count = inp_w * inp_h * 3 / 2;
    const int enc_count = inp_count / 2;

    uint8_t *y_ptr = inp_yuv420nv21;
    uint8_t *v_ptr = y_ptr + inp_w * inp_h;
    uint8_t *u_ptr = v_ptr + 1;

    image_data_t image_data;
    image_data.type = YUV_DATA_T;
    image_data.data.yuv.planes[0] = y_ptr;
    image_data.data.yuv.planes[1] = u_ptr;
    image_data.data.yuv.planes[2] = v_ptr;
    image_data.data.yuv.pixel_strides[0] = 1;
    image_data.data.yuv.pixel_strides[1] = 2;
    image_data.data.yuv.pixel_strides[2] = 2;
    image_data.data.yuv.row_strides[0] = inp_w;
    image_data.data.yuv.row_strides[1] = inp_w;
    image_data.data.yuv.row_strides[2] = inp_w;

    /* Init */
    int fd =
        open("profile.csv", O_WRONLY | O_CREAT | O_APPEND, S_IWRITE | S_IREAD);

    if (fd == -1) {
        perror("Cannot open file");
        return 1;
    }

    std::vector<std::string> source_files = {
        "../../android/app/src/main/assets/kernels/copy.cl"};
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
    codec_config_t codec_config;
    codec_config.rotation = rotation;
    codec_config.do_segment = do_segment;
    codec_config.compression_type = compression_type;
    codec_config.device_type = device_index;
    codec_config.config.jpeg.quality = quality;

    pocl_image_processor_context *ctx;

    status = create_pocl_image_processor_context(
        &ctx, 2, inp_w, inp_h, config_flags, codec_sources.at(0).c_str(),
        codec_sources.at(0).size(), fd);
    assert(status == CL_SUCCESS);

    // create read thread
    std::thread thread(read_function, ctx);

    /* Process */
    while (frame_index < NFRAMES) {
        int64_t image_timestamp = get_timestamp_ns();
        auto since_last_frame_sec =
            (float)(image_timestamp - last_image_timestamp) / 1e9f;

        if (since_last_frame_sec < (1.0f / FPS)) {
            continue;
        }

        FrameMark;

        if (dequeue_spot(ctx, 1000) != 0) {
            continue;
        }

        last_image_timestamp = image_timestamp;
        image_data.image_timestamp = image_timestamp;

        status = submit_image(ctx, codec_config, image_data, is_eval_frame);
        assert(status == CL_SUCCESS);

        frame_index += 1;
    }

    // join the thread
    thread.join();

    status = destroy_pocl_image_processor_context(&ctx);
    assert(status == CL_SUCCESS);

    close(fd);
    free(inp_yuv420nv21);

    return status;
}
