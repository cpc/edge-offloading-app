#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "fcntl.h"
#include "sys/stat.h"
#include "unistd.h"

// #include <rename_opencl.h>
#include <CL/cl.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "opencl_utils.hpp"
#include "poclImageProcessor.h"
#include "stb_image.h"

#define MAX_DETECTIONS 10
#define MASK_W 160
#define MASK_H 120

static int DETECTION_COUNT = 1 + MAX_DETECTIONS * 6;
static int SEGMENTATION_COUNT = MAX_DETECTIONS * MASK_W * MASK_H;
static int SEG_POSTPROCESS_COUNT = 4 * MASK_W * MASK_H; // RGBA image

int main() {
    int platform_id = 0;
    // int enc_device_id = 0; // on my machine pthread device
    // int dec_device_id = 0; // on my machine pthread device
    // int dnn_device_id = 0; // on my machine basic device

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

    /* Settings */
    const int config_flags = ENABLE_PROFILING | YUV_COMPRESSION;

    const int device_index = 0;
    const int do_segment = 1;
    const compression_t compression_type = YUV_COMPRESSION;
    const int quality = 80;
    const int rotation = 0;

    /* Read input image */
    std::string inp_name = "../android/app/src/main/assets/bus_640x480.jpg";

    int inp_w, inp_h, nch;
    uint8_t *inp_pixels = stbi_load(inp_name.data(), &inp_w, &inp_h, &nch, 3);
    throw_if_nullptr(inp_pixels, "Error opening input file");
    printf("OpenCL: Opened file '%s', %dx%d\n", inp_name.c_str(), inp_w, inp_h);

    // Convert inp image to YUV420 (U/V planes separate)
    cv::Mat inp_img(inp_h, inp_w, CV_8UC3, (void *)(inp_pixels));
    cv::cvtColor(inp_img, inp_img, cv::COLOR_RGB2YUV_YV12);
    // TODO: Remove this debug loop:
    for (int i = 0; i < inp_w * inp_h; ++i) {
        inp_img.data[i] = inp_pixels[3 * i];
    }
    cv::Mat inp_gray(inp_h, inp_w, CV_8UC1, (void *)(inp_img.data));
    cv::imwrite("inp_r.png", inp_gray);
    const int inp_count = inp_w * inp_h * 3 / 2;
    const int enc_count = inp_count / 2;
    const int yrow_stride = inp_w;
    const int uvrow_stride = inp_w;
    const int ypixel_stride = 1;
    const int uvpixel_stride = 2;
    const uint8_t *y_ptr = inp_img.data;
    const uint8_t *v_ptr = y_ptr + inp_w * inp_h;
    const uint8_t *u_ptr = v_ptr + 1;

    std::vector<int32_t> detections(DETECTION_COUNT);
    std::vector<uint8_t> segmentations(SEG_POSTPROCESS_COUNT);

    /* Init */
    int fd =
        open("profile.csv", O_WRONLY | O_CREAT | O_APPEND, S_IWRITE | S_IREAD);

    if (fd == -1) {
        perror("Cannot open file");
        return 1;
    }

    std::vector<std::string> source_files = {
        "../android/app/src/assets/kernels/copy.cl"};
    auto codec_sources = read_files(source_files);

    status = initPoclImageProcessor(inp_w, inp_h, config_flags,
                                    codec_sources.at(0).c_str(),
                                    codec_sources.at(0).size(), fd);

    /* Process */
    status = poclProcessYUVImage(
        device_index, do_segment, compression_type, quality, rotation, y_ptr,
        yrow_stride, ypixel_stride, u_ptr, v_ptr, uvrow_stride, uvpixel_stride,
        detections.data(), segmentations.data());

    close(fd);

    destroy_pocl_image_processor();

    return status;
}
