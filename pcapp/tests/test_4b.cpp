//
// Created by rabijl on 3.7.2024.
//
#include "dnn_stage.h"
#include "opencl_utils.hpp"
#include "rename_opencl.h"
#include "segment_4b_compression.hpp"
#include "sharedUtils.h"
#include <CL/cl.h>
#include <cassert>
#include <string>
#include <vector>

// uncomment these for some debug output
//#define PRINT_COMPRESS_BUF
//#define PRINT_SEGMENT_DATA

int main() {
    cl_int status;

    cl_platform_id platform_id;

    status = clGetPlatformIDs(1, &platform_id, NULL);
    CHECK_AND_RETURN(status, "can't get platform id");

    cl_uint dev_count = 0;

    status =
        clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &dev_count);
    CHECK_AND_RETURN(status, "can't get device count");
    assert(dev_count != 0);
    cl_device_id device_ids[dev_count];

    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, dev_count,
                            device_ids, NULL);
    CHECK_AND_RETURN(status, "can't get device id");

    cl_context context =
        clCreateContext(nullptr, dev_count, device_ids, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "could not create context");

    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES,
                                                CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(
        context, device_ids[0], properties, &status);

    std::vector<std::string> source_files= {"../../../android/app/src/main/assets/kernels/compress_seg.cl"};
    auto source_strings = read_files(source_files);
    assert(source_strings[0].length() > 0 && "could not open files");

    char const *source_string = source_strings.at(0).c_str();
    size_t const source_size = source_strings.at(0).size();

    cl_device_id segment_devs[] = {device_ids[0], device_ids[0]};
    segment_4b_context_t *segment_ctx =
        init_segment_4b(context, queue, queue, segment_devs, MASK_SZ1, MASK_SZ2,
                        source_size, source_string, &status);
    CHECK_AND_RETURN(status, "could not setup segment 4b context");

    size_t segment_size;
    unsigned char * const segment_data = (unsigned char *const)
        read_bin_file("data/segmentation.bin", &segment_size);
    assert(segment_size == MASK_SZ1 * MASK_SZ2 * sizeof(cl_uchar));
    if(nullptr == segment_data){
        exit(-2);
    }

    cl_mem segment_buf = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                        segment_size, nullptr, &status);
    CHECK_AND_RETURN(status, "could not create segment buffer");
    clEnqueueWriteBuffer(queue, segment_buf, CL_TRUE, 0, segment_size,
                        segment_data, 0, NULL, NULL);

    size_t detect_size;
    int *detect_data = (int *)
        read_bin_file("data/detections.bin", &detect_size);
    assert(detect_size == DET_COUNT*sizeof(cl_int));
    if(nullptr == detect_data){
        exit(-2);
    }
    printf("number of detections: %d\n", detect_data[0]);
    for(int i =0; i < detect_data[0]; i++) {
        printf("%d,", detect_data[1 + 6*i]);
    }
    printf("\n");

    cl_mem detect_buf = clCreateBuffer(context, CL_MEM_READ_ONLY, detect_size,
                                       nullptr, &status);
    CHECK_AND_RETURN(status, "could not create detect buffer");
    cl_event read_event;
    clEnqueueWriteBuffer(queue, detect_buf, CL_TRUE, 0, detect_size, detect_data,
                        0, NULL, &read_event);
    free(detect_data);

    cl_mem output_buf =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY, segment_size, NULL, &status);

    event_array_t *event_array = create_event_array_pointer(10);

    cl_event encode_event;
    encode_segment_4b(segment_ctx, &read_event, segment_buf, detect_buf,
                      output_buf, event_array, &encode_event);

    unsigned char *const result_array =
        (unsigned char *const)(malloc(segment_size));
    clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, segment_size,
                        result_array, 1, &encode_event, NULL);

    clFinish(queue);

#ifdef PRINT_COMPRESS_BUF
    size_t compress_buf_size = (MASK_H * MASK_W) / 2 * sizeof(cl_uchar);
    cl_uchar *const compress_buf = (cl_uchar *const)malloc(compress_buf_size);
    clEnqueueReadBuffer(queue, segment_ctx->compress_buf, CL_TRUE, 0,
                        compress_buf_size, compress_buf, 1, &encode_event,
                        NULL);
    printf("compress buf: \n");
    for (int i = 0; i < compress_buf_size; i++) {
        if (i % (MASK_W / 2) == 0) {
            printf("\n");
        }
        printf("%u,", compress_buf[i]);
    }
    printf("\n");
    free(compress_buf);
#endif

#ifdef PRINT_SEGMENT_DATA
    printf("segment data: \n");
    for (int i = 0; i < segment_size; i++) {
        if (i % 160 == 0) {
            printf("\n");
        }
        printf("%hu,", segment_data[i]);
    }
    printf("\n");
#endif

    bool correct = true;
    for (int i = 0; i < segment_size; i++) {
        if (result_array[i] != segment_data[i]) {
            printf("reconstructed results do not match source \n");
            printf("at index: %d, input: %u, output: %u\n", i, segment_data[i],
                   result_array[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("results match!");
    }

    clReleaseMemObject(segment_buf);
    clReleaseMemObject(detect_buf);
    clReleaseMemObject(output_buf);
    clReleaseCommandQueue(queue);

    destroy_segment_4b(&segment_ctx);
    free_event_array(event_array);

    free(result_array);
    free(segment_data);
    return 0;
}