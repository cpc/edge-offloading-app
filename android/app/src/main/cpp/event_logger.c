//
// Created by rabijl on 8/28/23.
// implementations for event_logger.h
//
#include "event_logger.h"
#include "sharedUtils.h"
#include <assert.h>

event_array_t
create_event_array(const int size) {
    event_pair_t *array = (event_pair_t *) malloc(size * sizeof(event_pair_t));
    event_array_t res = {size, 0, array};
    return res;
}

void
reset_event_array(event_array_t *array) {
    array->current_capacity = 0;
}

void
free_event_array(event_array_t *array) {
    free(array->array);
}

void
append_to_event_array(event_array_t *array, cl_event event, const char *description) {
    assert(array->max_capacity > array->current_capacity);

    array->array[array->current_capacity] = (event_pair_t) {event, description};
    array->current_capacity += 1;
}

int
print_events(int fd, int frame_index, const event_array_t *array) {
    int status;
    cl_ulong event_time;
    int ret_status = CL_SUCCESS;

    for (int i = 0; i < array->current_capacity; i++) {
        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_QUEUED,
                                         sizeof(cl_ulong),
                                         &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("ERROR: could not read queued time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,queued_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_SUBMIT,
                                         sizeof(cl_ulong),
                                         &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("ERROR: could not read submit time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,submit_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong),
                                         &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("ERROR: could not read start time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,start_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong),
                                         &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("ERROR: could not read end time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,end_ns,%lu\n", frame_index, array->array[i].description, event_time);

#ifdef PRINT_PROFILE_TIME
        PRINT_DIFF(event_time - start, array->array[i].description);
#endif

    }

    return ret_status;
}

void
release_events(event_array_t *array) {
    for (int i = 0; i < array->current_capacity; i++) {
        clReleaseEvent(array->array[i].event);
    }
}
