//
// Created by rabijl on 16.8.2024.
//

#include "PingThread.h"
#include "platform.h"
#include <array>
#include <cassert>

using namespace std;

PingThread::PingThread(cl_context context) {

    pingCtx = nullptr;
    ping_fillbuffer_init(&(pingCtx), context);
    lastPing = -1;

    stopThread = false;
    workAvailable = false;
    queue = nullptr;
    workThread = thread(&PingThread::threadFunction, this);

}

void PingThread::threadFunction() {

    cl_event migrateEvent;
    cl_int status;
    cl_command_queue threadQueue;
    constexpr uint8_t pattern[1] = {255};
    constexpr auto size = std::size(pattern);
    cl_ulong start_time_ns, end_time_ns;

    while (true) {
        {
            std::unique_lock lock(conditionMutex);
            // don't wait on condition if work is available or it's time to stop
            workCondition.wait(lock, [this] { return workAvailable || stopThread; });
            threadQueue = queue;
            workAvailable = false;
            if (stopThread) {
                return;
            }
        }

        status = clEnqueueMigrateMemObjects(threadQueue, 1, &(pingCtx->buf),
                                            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr,
                                            &migrateEvent);
        if (status != CL_SUCCESS) {
            LOGW("could not migrate ping buffer");
            continue;
        }
        status = clEnqueueFillBuffer(threadQueue, pingCtx->buf, &pattern, size, 0, size, 1,
                                     &migrateEvent,
                                     &(pingCtx->event));
        if (status != CL_SUCCESS) {
            clReleaseEvent(migrateEvent);
            LOGW("could not enqueue ping buffer");
            continue;
        }

        clWaitForEvents(1, &(pingCtx->event));

        status = clGetEventProfilingInfo(pingCtx->event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &start_time_ns, NULL);
        if (status != CL_SUCCESS) {
            LOGW("could not get ping start");
            // TODO: remove assert after done testing
            assert(0);
            goto FINISH;
        }

        status = clGetEventProfilingInfo(pingCtx->event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong),
                                         &end_time_ns, NULL);
        if (status != CL_SUCCESS) {
            LOGW("could not get ping end");
            // TODO: remove assert after done testing
            assert(0);
            goto FINISH;
        }

        {
            unique_lock lock(conditionMutex);
            lastPing = (int64_t)(end_time_ns - start_time_ns);
        }

        FINISH:
        clReleaseEvent(migrateEvent);
        clReleaseEvent(pingCtx->event);
    }

}

PingThread::~PingThread() {

    {
        std::unique_lock lock(conditionMutex);
        stopThread = true;
        workCondition.notify_one();
    }

    workThread.join();
    ping_fillbuffer_destroy(&pingCtx);
}

int64_t PingThread::getPing() {
    std::unique_lock lock(conditionMutex);
    int64_t ret = lastPing;
    lastPing = -1;
    return ret;
}

void PingThread::ping(cl_command_queue device_queue) {
    std::unique_lock lock(conditionMutex);
    queue = device_queue;
    workAvailable = true;
    workCondition.notify_one();

}
