//
// Created by rabijl on 16.7.2024.
//
#include "RawImageReader.hpp"
#include "poclImageProcessorUtils.h"
#include <cstdlib>
#include <fcntl.h>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

using namespace std;
using namespace cv;
int main() {

    const char *vidPath = getenv("VIDPATH");
    std::cout << "vidpath: " << vidPath << std::endl;

    int fd = open(vidPath, O_RDONLY);
    if (fd == -1) {
        cout << "open error: " << strerror(errno);
        cout << " try setting VIDPATH to a raw yuv image \n";
        exit(-1);
    }

    int width = 640;
    int height = 480;

    RawImageReader reader = RawImageReader(width, height, fd);

    image_data_t input_image;
    reader.readImage(&input_image);

    auto *yuv_buf = new uint8_t[width * height * 3 / 2];

    // convert image to semi_planaer nv21, similar to what happens in the pipv2
    copy_yuv_to_arrayV2(width, height, input_image, NO_COMPRESSION, yuv_buf);

    // convert to rgb image similar to what happens in run_onnx_inference
    Mat yuvMat = Mat(height * 3 / 2, width, CV_8UC1, yuv_buf);

    Mat rgbMat;
    cvtColor(yuvMat, rgbMat, COLOR_YUV2RGB_NV21);

    imshow("nv21 image", rgbMat);

    waitKey();

    // test that the last frame is properly set
    int lastFrame;
    for (int i = 1; i < reader.getTotalFrames(); i++) {
        lastFrame = reader.readImage(&input_image);
    }
    assert(lastFrame);

    delete[] yuv_buf;
    close(fd);
}