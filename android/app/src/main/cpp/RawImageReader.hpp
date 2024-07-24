//
// Created by rabijl on 12.7.2024.
//

#ifndef POCL_AISA_DEMO_RAWIMAGEREADER_HPP
#define POCL_AISA_DEMO_RAWIMAGEREADER_HPP

#include "platform.h"
#include "poclImageProcessorTypes.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <unistd.h>

class RawImageReader {
public:
    RawImageReader(int width, int height, int fd) noexcept(false);

    int readImage(image_data_t *image) noexcept(false);

    [[nodiscard]] int getTotalFrames() const;

    void reset();

    ~RawImageReader();

private:
    FILE *file;
    int width;
    int height;
    int total_frames;
    int current_frame;
    int image_size;
    uint8_t *image_buf;
};

void openFile();

#endif // POCL_AISA_DEMO_RAWIMAGEREADER_HPP
