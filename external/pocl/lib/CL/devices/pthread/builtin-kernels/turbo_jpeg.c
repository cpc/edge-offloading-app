//
// Created by rabijl on 16.8.2023.
//
#include "turbo_jpeg.h"
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>

#ifdef TRACY_ENABLE
#include <TracyC.h>
#endif

//uncomment this to save images
//#define SAVE_IMAGES

/* turbojpeg reference:
 * https://rawcdn.githack.com/libjpeg-turbo/libjpeg-turbo/main/doc/html/group___turbo_j_p_e_g.html
 */

static tjhandle tjDecompressHandle = NULL;
static tjhandle tjCompressHandle = NULL;
static uint8_t *yuv420_planar_buf = NULL;
// TODO: destroy mutex when done. currently quite hard to determine when to do this.
static pthread_mutex_t compress_kernel_lock = PTHREAD_MUTEX_INITIALIZER;

// tj3CompressFromYUV8 will reallocate the output buffer and we definitely
// don't want that to happen if that pointer is pointing to our ocl buffer.
// so keep an internal jpeg buffer.
unsigned char *internalJpegBuf = NULL;
// used to keep track of how big the internalJpegBuf is
size_t internalJpegSize;

/**
 * function to write the jpeg buffers to a file
 * @param dir the target directory to store images in
 * @param jpeg the buffer containing the image
 * @param jpeg_size the size of the buffer
 */
static void save_image(const char * const dir, uint8_t const * const jpeg, const uint64_t jpeg_size) {

    static const char format[] = "%s/%ld.jpg";
    char file_name[22 + sizeof format + strlen(dir)];
    snprintf(file_name, sizeof file_name, format, dir, jpeg_size );
    FILE * file = fopen(file_name, "w");
    if(NULL == file) {
        POCL_MSG_ERR("could not open file to write jpeg to\n");
        return;
    }
    fwrite(jpeg, 1, jpeg_size,file);
    fclose(file);
}

void _pocl_kernel_pocl_compress_to_jpeg_yuv420nv21_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z)
{
#ifdef TRACY_ENABLE
    TracyCZone (ctx, 1);
#endif
    void **arguments = *(void ***)(args);
    void **arguments2 = (void **)(args);

    int nargs = 0;
    const uint8_t *input = (const uint8_t *)(arguments[nargs++]);
    int32_t width = *(int32_t*)(arguments2[nargs++]);
    int32_t height = *(int32_t*)(arguments2[nargs++]);
    int32_t quality = *(int32_t*)(arguments2[nargs++]);
    uint8_t *output = (uint8_t *)(arguments[nargs++]);
    uint64_t *output_size = (uint64_t *)(arguments[nargs++]);

    turbo_jpeg_run_compress_to_jpeg_yuv420nv21(input, width, height, quality,
                                               output, output_size);
#ifdef TRACY_ENABLE
    TracyCZoneEnd (ctx);
#endif
}

void _pocl_kernel_pocl_decompress_from_jpeg_rgb888_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z)
{
#ifdef TRACY_ENABLE
    TracyCZone (ctx, 1);
#endif
    void **arguments = *(void ***)(args);
    void **arguments2 = (void **)(args);

    int nargs = 0;
    const uint8_t *input = (const uint8_t *)(arguments[nargs++]);
    const uint64_t *input_size = (const uint64_t *)(arguments[nargs++]);
    int32_t width = *(int32_t*)(arguments2[nargs++]);
    int32_t height = *(int32_t*)(arguments2[nargs++]);
    uint8_t *output = (uint8_t *)(arguments[nargs++]);

    turbo_jpeg_run_decompress_from_jpeg_rgb888(input, input_size, width, height,
                                               output);
#ifdef TRACY_ENABLE
    TracyCZoneEnd (ctx);
#endif
}

int32_t
turbo_jpeg_run_decompress_from_jpeg_rgb888 (const uint8_t *input,
                                            const uint64_t *input_size,
                                            int32_t width, int32_t height,
                                            uint8_t *output)
{
  // status = tj3Set(tjCompressHandle, TJPARAM_FASTUPSAMPLE, 1);
  // assert(status && "JPEG: Setting fast upsampling");
  assert(tjDecompressHandle && "tjDecompressHandle has not been initialized");

  int ret = tj3DecompressHeader (tjDecompressHandle, input, *input_size);
  if (ret == 0)
    {

#ifdef SAVE_IMAGES
      save_image(".", input, *input_size);
#endif
      ret = tj3Decompress8 (tjDecompressHandle, input, *input_size, output, 0,
                            TJCS_RGB);
      if (ret != 0)
        {
          POCL_MSG_ERR ("Decompression did not go well: %s\n",
                        tj3GetErrorStr (tjDecompressHandle));
        }
    }
  else
    {
      POCL_MSG_ERR ("Failed getting headers: %s\n",
                    tj3GetErrorStr (tjDecompressHandle));
    }
}


int32_t
turbo_jpeg_run_compress_to_jpeg_yuv420nv21(const uint8_t *input,
                                           int32_t width,
                                           int32_t height,
                                           int32_t quality,
                                           uint8_t *output,
                                           uint64_t *output_size)
{
    pthread_mutex_lock(&compress_kernel_lock);
  if (yuv420_planar_buf == NULL)
  {
    yuv420_planar_buf = (uint8_t *)(malloc(width * height * 3 / 2));
  }
  assert(tjCompressHandle && "tjCompresshandle has not been initialized");

  //TODO: look into not having to make a copy and reordering data

  // Convert interleaved U/V planes into separate U/V planes
  memcpy(yuv420_planar_buf, input, width * height); // Y
  for (int i = 0; i < width * height / 4; i += 1)
  {
    yuv420_planar_buf[width * height + i] = input[width * height + 2 * i];                          // U or V
      yuv420_planar_buf[width * height + width * height / 4 + i] = input[width * height + 2 * i + 1]; // V or U
  }

    // Encode the planar YUV 420-subsampled image as JPEG
    int status = 0;

    status = tj3Set(tjCompressHandle, TJPARAM_SUBSAMP, TJSAMP_420);
    assert((status == 0) && "JPEG: Setting subsampling factor");

    status = tj3Set(tjCompressHandle, TJPARAM_QUALITY, quality);
    assert((status == 0) && "JPEG: Setting subsampling factor");

    // status = tj3Set(tjCompressHandle, TJPARAM_FASTDCT, 1);
    // assert(status && "JPEG: Setting fast DCT");

    const int align = 1;
    // TODO: look into configuring the compresshandle with norealloc and setting output_size to the size of the buffer
    // the first time
    status = tj3CompressFromYUV8(tjCompressHandle, yuv420_planar_buf, width, align, height, &internalJpegBuf , &internalJpegSize);
    assert((status == 0) && "JPEG: Compressing");
    memcpy(output, internalJpegBuf, internalJpegSize);
    *output_size = internalJpegSize;

#ifdef SAVE_IMAGES
    // the download dir of android
    save_image("/storage/self/primary/Download", output, *output_size);
#endif

    pthread_mutex_unlock(&compress_kernel_lock);
}

void
turbo_jpeg_run_init_decompress_jpeg_handle_rgb888(uint8_t *const ctx_handle) {

    tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
    assert(handle && "tjInitDecompress failed");
    memcpy(ctx_handle, &handle, sizeof(tjhandle));

}

void
turbo_jpeg_run_destroy_decompress_jpeg_handle_rgb888(uint8_t *const ctx_handle) {

    tjhandle *handle = (tjhandle) ctx_handle;
    if (NULL != *handle) {
        tj3Destroy(*handle);
        *handle = NULL;
    } else {
        POCL_MSG_ERR ("passed an empty handle to destroy.decompress.jpeg.handle.rgb88\n");
    }
}

void
turbo_jpeg_run_decompress_from_jpeg_handle_rgb888(const uint8_t *ctx_handle,
                                                  const uint8_t *input,
                                                  const uint64_t *input_size,
                                                  uint8_t *output) {

    tjhandle handle;
    memcpy(&handle, ctx_handle, sizeof(tjhandle));

    assert(handle && "tjDecompressHandle has not been initialized");

    int ret = tj3DecompressHeader(handle, input, *input_size);
    if (ret == 0) {

#ifdef SAVE_IMAGES
        save_image(".", input, *input_size);
#endif

        ret = tj3Decompress8(handle, input, *input_size, output, 0,
                             TJCS_RGB);
        if (ret != 0){
            POCL_MSG_ERR ("Decompression did not go well: %s\n",
                          tj3GetErrorStr (tjDecompressHandle));
        }
    } else {
        POCL_MSG_ERR ("Failed getting headers: %s\n",
                      tj3GetErrorStr(handle));
    }
}

void _pocl_kernel_pocl_init_decompress_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

#ifdef TRACY_ENABLE
    TracyCZone (ctx, 1);
#endif
    void **arguments = *(void ***) (args);

    uint8_t *const buffer = (uint8_t *const) (arguments[0]);

    turbo_jpeg_run_init_decompress_jpeg_handle_rgb888(buffer);
#ifdef TRACY_ENABLE
    TracyCZoneEnd (ctx);
#endif
}

void _pocl_kernel_pocl_destroy_decompress_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

#ifdef TRACY_ENABLE
    TracyCZone (ctx, 1);
#endif
    void **arguments = *(void ***) (args);

    uint8_t *const buffer = (uint8_t *const) (arguments[0]);

    turbo_jpeg_run_destroy_decompress_jpeg_handle_rgb888(buffer);
#ifdef TRACY_ENABLE
    TracyCZoneEnd (ctx);
#endif
}


void _pocl_kernel_pocl_decompress_from_jpeg_handle_rgb888_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {
    
#ifdef TRACY_ENABLE
    TracyCZone (ctx, 1);
#endif
    void **arguments = *(void ***) (args);

    uint8_t *const buffer = (uint8_t *const) (arguments[0]);
    const uint8_t *input = (const uint8_t *) (arguments[1]);
    uint64_t *const input_size = (uint64_t *const) (arguments[2]);
    uint8_t *const output = (uint8_t *const) (arguments[3]);

    turbo_jpeg_run_decompress_from_jpeg_handle_rgb888(buffer, input, input_size, output);
#ifdef TRACY_ENABLE
    TracyCZoneEnd (ctx);
#endif
}

void init_turbo_jpeg(cl_program program, cl_uint device_i) {
    if (NULL == tjCompressHandle) {
        tjCompressHandle = tj3Init(TJINIT_COMPRESS);

    }
    assert(tjCompressHandle && "tjInitCompress failed\n");

    if (NULL == tjDecompressHandle) {
        tjDecompressHandle = tj3Init(TJINIT_DECOMPRESS);
    }
    assert(tjDecompressHandle && "tjInitDecompress failed\n");

    yuv420_planar_buf = NULL;
}

void destroy_turbo_jpeg(cl_device_id device, cl_program program,
                        unsigned dev_i)
{
    if(NULL != tjCompressHandle) {
        tj3Destroy(tjCompressHandle);
        tjCompressHandle = NULL;
    }
    if(NULL != tjDecompressHandle) {
        tj3Destroy(tjDecompressHandle);
        tjDecompressHandle = NULL;
    }
    if(NULL != internalJpegBuf) {
        tj3Free(internalJpegBuf);
        internalJpegBuf = NULL;
    }

  if (yuv420_planar_buf != NULL)
  {
    free(yuv420_planar_buf);
    yuv420_planar_buf = NULL;
  }
}
