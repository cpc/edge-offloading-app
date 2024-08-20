//
// Created by rabijl on 16.8.2024.
//

#ifndef POCL_AISA_DEMO_PINGTHREAD_H
#define POCL_AISA_DEMO_PINGTHREAD_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "rename_opencl.h"
#include <CL/cl.h>
#include <CL/cl_ext_pocl.h>
#include "event_logger.h"
#include "testapps.h"
#include <thread>
#include <condition_variable>

class PingThread {

public:
    explicit PingThread(cl_context context);

    /**
     * enqueue a ping to a device that belongs to the given queue
     * @param device_queue queue of device to ping
     */
    void ping(cl_command_queue device_queue);

    /**
     * returns the ping of the finished ping command.
     * @return the ping in ns or -1 if ping is already collected or not done yet
     */
    int64_t getPing();

    ~PingThread();

private:
    cl_command_queue queue;
    ping_fillbuffer_context_t *pingCtx;
    int64_t lastPing;

    std::thread workThread;
    std::condition_variable workCondition;
    std::mutex conditionMutex;
    bool stopThread;
    bool workAvailable;
    /**
     * function running in the ping thread
     */
    void threadFunction();

};

#endif //POCL_AISA_DEMO_PINGTHREAD_H

