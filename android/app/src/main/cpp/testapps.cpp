#include "rename_opencl.h"
#include <CL/cl.h>
#include <CL/cl_ext_pocl.h>

#include <assert.h>

#include "testapps.h"
#include "platform.h"
#include "poclImageProcessorV2.h"
#include "sharedUtils.h"

int test_vec_add() {
    cl_platform_id platform;
    cl_device_id devices[MAX_NUM_CL_DEVICES] = {NULL};
    cl_uint devices_found;
    cl_int status;

    LOGI("<<< TMP Start <<<\n");

    status = clGetPlatformIDs(1, &platform, NULL);
    CHECK_AND_RETURN(status, "TMP getting platform id failed");

    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, MAX_NUM_CL_DEVICES, devices,
                            &devices_found);
    CHECK_AND_RETURN(status, "TMP getting device id failed");
    LOGI("TMP Platform has %d devices\n", devices_found);
    assert(devices_found > 0);

    char result_array[256];
    for (unsigned i = 0; i < devices_found; ++i) {
        clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 256 * sizeof(char), result_array, NULL);
        LOGI("TMP device %d: CL_DEVICE_NAME:    %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DEVICE_VERSION, 256 * sizeof(char), result_array, NULL);
        LOGI("TMP device %d: CL_DEVICE_VERSION: %s\n", i, result_array);
        clGetDeviceInfo(devices[i], CL_DRIVER_VERSION, 256 * sizeof(char), result_array, NULL);
        LOGI("TMP device %d: CL_DRIVER_VERSION: %s\n", i, result_array);
    }

    cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                   0};

    cl_context tmp_context = clCreateContext(cps, devices_found, devices, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "TMP creating context failed");
    LOGI("TMP built context\n");

    cl_program tmp_program = clCreateProgramWithBuiltInKernels(tmp_context, 1, devices,
//                                                               "pocl.add.i8;pocl.dnn.detection.u8;pocl.dnn.segmentation.postprocess.u8;pocl.dnn.segmentation.reconstruct.u8;pocl.dnn.eval.iou.f32",
                                                               "pocl.add.i8",
                                                               &status);
    CHECK_AND_RETURN(status, "TMP creation of program failed");
    LOGI("TMP built kernels\n");

    status = clBuildProgram(tmp_program, 1, devices, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP building of program failed");
    LOGI("TMP Created and built program");

    cl_command_queue_properties cq_properties[] = {0};
    cl_command_queue tmp_command_queue = clCreateCommandQueueWithProperties(tmp_context, devices[0],
                                                                            cq_properties,
                                                                            &status);
    CHECK_AND_RETURN(status, "TMP creating eval command queue failed");
    LOGI("TMP Created CQ\n");

    cl_kernel tmp_kernel = clCreateKernel(tmp_program, "pocl.add.i8", &status);
    CHECK_AND_RETURN(status, "TMP creating eval kernel failed");
    LOGI("TMP Created kernel\n");

    char A[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    char B[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    char C[8] = {0};

    cl_mem tmp_buf_a = clCreateBuffer(tmp_context, CL_MEM_READ_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer A");
    cl_mem tmp_buf_b = clCreateBuffer(tmp_context, CL_MEM_READ_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer B");
    cl_mem tmp_buf_c = clCreateBuffer(tmp_context, CL_MEM_WRITE_ONLY, 8, NULL, &status);
    CHECK_AND_RETURN(status, "TMP failed to create buffer C");
    LOGI("TMP Created buffers\n");

    status = clSetKernelArg(tmp_kernel, 0, sizeof(cl_mem), &tmp_buf_a);
    status |= clSetKernelArg(tmp_kernel, 1, sizeof(cl_mem), &tmp_buf_b);
    status |= clSetKernelArg(tmp_kernel, 2, sizeof(cl_mem), &tmp_buf_c);
    CHECK_AND_RETURN(status, "TMP could not assign kernel args");
    LOGI("TMP Set kernel args\n");

    status = clEnqueueWriteBuffer(tmp_command_queue, tmp_buf_a, CL_TRUE, 0, 8, A, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to write A buffer");
    status = clEnqueueWriteBuffer(tmp_command_queue, tmp_buf_b, CL_TRUE, 0, 8, B, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to write B buffer");
    LOGI("TMP Wrote buffers\n");

    size_t global_size = 1;
    size_t local_size = 1;
    status = clEnqueueNDRangeKernel(tmp_command_queue, tmp_kernel, 1, NULL, &global_size,
                                    &local_size, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to enqueue ND range kernel");
    LOGI("TMP Enqueued kernel\n");

    status = clEnqueueReadBuffer(tmp_command_queue, tmp_buf_c, CL_TRUE, 0, 8, C, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "TMP failed to read C buffer");
    LOGI("TMP Read buffers\n");

    for (int i = 0; i < 8; ++i) {
        LOGI("TMP C[%d]: %d\n", i, C[i]);
        if (C[i] != 3) {
            LOGE("TMP: Wrong result!\n");
            return -1;
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
    status = clReleaseKernel(tmp_kernel);
    CHECK_AND_RETURN(status, "TMP failed to release kernel");
    status = clReleaseProgram(tmp_program);
    CHECK_AND_RETURN(status, "TMP failed to release program");
    status = clReleaseDevice(devices[0]);
    CHECK_AND_RETURN(status, "TMP failed to release device");
    status = clReleaseContext(tmp_context);
    CHECK_AND_RETURN(status, "TMP failed to release context");

    LOGI(">>>>> TMP SUCCESS <<<<<<\n");

    return CL_SUCCESS;
}

int ping_fillbuffer_init(ping_fillbuffer_context_t **out_ctx, cl_context context) {
    cl_int status = CL_SUCCESS;

    ping_fillbuffer_context_t *ctx = (ping_fillbuffer_context_t *) malloc(
            sizeof(ping_fillbuffer_context_t));

    size_t size = 1;
    ctx->buf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &status);
    CHECK_AND_RETURN(status, "PING failed to create buffer");

    *out_ctx = ctx;

    return status;
}

int ping_fillbuffer_run(ping_fillbuffer_context_t *ctx, cl_command_queue queue,
                        event_array_t *event_array) {
    cl_int status = CL_SUCCESS;

    uint8_t pattern[] = {255};
    size_t pattern_size = sizeof(uint8_t);
    size_t off = 0;
    size_t size = 1 * pattern_size;
    status = clEnqueueFillBuffer(queue, ctx->buf, &pattern, pattern_size, off, size, 0, NULL,
                                 &ctx->event);
    CHECK_AND_RETURN(status, "PING could not fill buffer");
//    append_to_event_array(event_array, ctx->event, VAR_NAME(fill_event));

//    status = clWaitForEvents(1, &ctx->event);
//    CHECK_AND_RETURN(status, "PING failed to wait for fill event");

    return status;
}

int ping_fillbuffer_destroy(ping_fillbuffer_context_t **ctx) {
    cl_int status = CL_SUCCESS;

    if (ctx != NULL && *ctx != NULL) {
        COND_REL_MEM((*ctx)->buf);
        free((*ctx)->buf);
    }

    return status;
}