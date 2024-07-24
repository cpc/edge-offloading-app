//
// Created by rabijl on 12.7.2024.
//

#include "RawImageReader.hpp"
#include <cstdlib>
#include <fstream>

#include <iostream>

using namespace cv;
using namespace std;

/**
 * Create a new raw image reader that creates a buffered reader
 * and a buffer to store temporary images in.
 * @param width of the image
 * @param height of the image
 * @param fd to the file, can be from android and will be duplicated.
 */
RawImageReader::RawImageReader(int width, int height, int fd) noexcept(false) {

    this->width = width;
    this->height = height;
    this->image_size = width * height * 3 / 2;
    this->current_frame = 0;

    // duplicate the filediscriptor since fclose will also close the fd
    // and the android side also want to close the fd.
    int dup_fd = dup(fd);
    if (dup_fd < 0) {
        throw invalid_argument(strerror(errno));
    }

    this->file = fdopen(dup_fd, "rb");
    if (file == nullptr) {
        throw invalid_argument(strerror(errno));
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    if (file_size % image_size != 0) {
        throw invalid_argument("file is not a multiple of image size");
    }
    this->total_frames = static_cast<int>(file_size / image_size);
    this->image_buf = new uint8_t[image_size];
}

RawImageReader::~RawImageReader() {
    fclose(file);
    delete[] image_buf;
}

/**
 * read an image from the file and wrap it in an image_data_t
 * @param image used to wrap raw image buffer
 * @return returns true if this is the last image
 */
int RawImageReader::readImage(image_data_t *image) noexcept(false) {

    if (current_frame >= total_frames) {
        throw out_of_range("reached end of file");
    }

    size_t bytesRead = fread(image_buf, 1, image_size, file);
    if (bytesRead != image_size) {
        throw runtime_error("could not read image");
    }
    current_frame++;

    image->type = YUV_DATA_T;
    yuv_image_data_t *yuv = &(image->data.yuv);
    // Y plane
    yuv->planes[0] = image_buf;
    // U plane
    yuv->planes[1] = &(image_buf[width * height]);
    // V plane
    // subsampled by 2x2 so the U plane is the size of height * width / 4
    yuv->planes[2] = &(image_buf[width * height * 5 / 4]);
    yuv->pixel_strides[0] = 1;
    yuv->pixel_strides[1] = 1;
    yuv->pixel_strides[2] = 1;
    yuv->row_strides[0] = width;
    yuv->row_strides[1] = width / 2;
    yuv->row_strides[2] = width / 2;

    return (current_frame == total_frames);
}

/**
 * @return the total number of frames of the file
 */
int RawImageReader::getTotalFrames() const { return total_frames; }

/**
 * set the reader back to the beginning of the file
 */
void RawImageReader::reset() {
    fseek(file, 0, SEEK_SET);
    current_frame = 0;
}
