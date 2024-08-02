//
// Created by rabijl on 1.8.2024.
//

#ifndef _JPEGREADER_H_
#define _JPEGREADER_H_

#include "poclImageProcessorTypes.h"
#include <memory>
#include <stdint.h>
#include <unistd.h>

class JPEGReader {
public:
  explicit JPEGReader(const char *path) noexcept(false);
  JPEGReader(int width, int height, const char *path) noexcept(false);
  bool readImage(image_data_t *image) noexcept(true);
  std::pair<int, int> getDimensions();

private:
  int width;
  int height;
  std::unique_ptr<uint8_t[]> image_buf;
  image_data_t image_data;

  void loadImage(const char *path, bool useImageDimensions);
};

#endif //_JPEGREADER_H_
