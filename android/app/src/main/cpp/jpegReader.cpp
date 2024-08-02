//
// Created by rabijl on 1.8.2024.
//

#include "jpegReader.h"
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <opencv2/imgproc.hpp>

JPEGReader::JPEGReader(int width, int height,
                       const char *path) noexcept(false) {
  this->width = width;
  this->height = height;

  loadImage(path, false);
}

JPEGReader::JPEGReader(const char *path) noexcept(false) {
  loadImage(path, true);
}

using namespace std;

void JPEGReader::loadImage(const char *path, bool useImageDimensions) {
  // gotten from the input image
  int inp_w, inp_h, nch;
  uint8_t *inp_pixels = stbi_load(path, &inp_w, &inp_h, &nch, 3);
  if (inp_pixels == nullptr) {
    throw invalid_argument("file can not be found \n");
  }
  if (useImageDimensions) {
    this->width = inp_w;
    this->height = inp_h;
  } else if ((inp_w != this->width || inp_h != this->height)) {
    const char format_string[] =
        "expected dimensions of (%d,%d), image is (%d,%d)";
    int maxLen = sizeof format_string + 4 * 10 + 1;
    char err_string[maxLen];
    snprintf(err_string, maxLen, format_string, this->width, this->height,
             inp_w, inp_h);
    throw invalid_argument(err_string);
  }

  // Convert inp image to YUV420 (U/V planes separate)
  cv::Mat inp_img(inp_h, inp_w, CV_8UC3, (void *)(inp_pixels));
  cv::cvtColor(inp_img, inp_img, cv::COLOR_RGB2YUV_YV12);

  // TODO: check whether this should be nv12 instead
  // Convert separate U/V planes into interleaved U/V planes
  image_buf = make_unique<uint8_t[]>(inp_w * inp_h * 3 / 2);
  memcpy(image_buf.get(), inp_img.data, inp_w * inp_h);

  for (int i = 0; i < inp_w * inp_h / 4; i += 1) {
    image_buf[inp_w * inp_h + 2 * i] =
        inp_img.data[inp_w * inp_h + inp_w * inp_h / 4 + i];
    image_buf[inp_w * inp_h + 2 * i + 1] = inp_img.data[inp_w * inp_h + i];
  }
  // copied into the image_buf, so can now be freed
  free(inp_pixels);

  const int inp_count = inp_w * inp_h * 3 / 2;
  const int enc_count = inp_count / 2;

  uint8_t *y_ptr = image_buf.get();
  uint8_t *v_ptr = y_ptr + inp_w * inp_h;
  uint8_t *u_ptr = v_ptr + 1;

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
}
bool JPEGReader::readImage(image_data_t *image) noexcept(true) {

  *image = this->image_data;
  return false;
}
std::pair<int, int> JPEGReader::getDimensions() { return {width, height}; }
