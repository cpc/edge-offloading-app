//
// Created by rabijl on 16.8.2024.
//
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "rename_opencl.h"
#include <CL/cl.h>

#include "PingThread.h"
#include "unistd.h"

#include "sharedUtils.h"
#include <cassert>

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

    for (int i = 0; i < dev_count; i++) {
        clReleaseDevice(device_ids[i]);
    }

    PingThread pingThread = PingThread(context);

    pingThread.ping(queue);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    auto ping = pingThread.getPing();
    assert(ping != -1);
    LOGI("ping time: %ld ns\n", ping);
    ping = pingThread.getPing();
    assert(ping == -1);
    LOGI("second ping time: %ld ns\n", ping);
    pingThread.ping(queue);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    ping = pingThread.getPing();
    assert(ping != -1);
    LOGI("ping time: %ld ns\n", ping);

    std::this_thread::sleep_for(std::chrono::milliseconds(5000));
    LOGI("done\n");

    clReleaseContext(context);
    clReleaseCommandQueue(queue);
}
