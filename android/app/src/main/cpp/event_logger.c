//
// Created by rabijl on 8/28/23.
// implementations for event_logger.h
//
#include "event_logger.h"
#include "sharedUtils.h"
#include <assert.h>
#include <string.h>

__attribute__ ((deprecated))
event_array_t create_event_array(const int size) {
    event_pair_t *array = (event_pair_t *) malloc(size * sizeof(event_pair_t));
    event_array_t res = {size, 0, array};
    return res;
}

event_array_t *create_event_array_pointer(const int size) {
    event_pair_t *array = (event_pair_t *) malloc(size * sizeof(event_pair_t));
    event_array_t *res = malloc(sizeof(event_array_t));
    res->array = array;
    res->max_capacity = size;
    res->current_capacity = 0;

    return res;
}

void free_event_array_pointer(event_array_t **array) {
    if (NULL == *array) {
        return;
    }
    free((*array)->array);
    free(*array);
    *array = NULL;
}

void reset_event_array(event_array_t *array) {
    array->current_capacity = 0;
}

__attribute__ ((deprecated))
void free_event_array(event_array_t *array) {
    if (NULL == array->array) {
        return;
    }
    free(array->array);
}

void append_to_event_array(event_array_t *array, cl_event event, const char *description) {
    assert(array->max_capacity > array->current_capacity);

    array->array[array->current_capacity] = (event_pair_t) {event, description};
    array->current_capacity += 1;
}

int log_events(int fd, int frame_index, const event_array_t *array) {
    int status;
    cl_ulong event_time;
    int ret_status = CL_SUCCESS;

    for (int i = 0; i < array->current_capacity; i++) {
        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_QUEUED,
                                         sizeof(cl_ulong), &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("EVENT LOGGER | Could not read queued time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,queued_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_SUBMIT,
                                         sizeof(cl_ulong), &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("EVENT LOGGER | Could not read submit time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,submit_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_START,
                                         sizeof(cl_ulong), &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("EVENT LOGGER | Could not read start time of %s at %s:%d returned with %d\n",
                 array->array[i].description, __FILE__, __LINE__, status);
            continue;
        }
        dprintf(fd, "%d,%s,start_ns,%lu\n", frame_index, array->array[i].description, event_time);

        status = clGetEventProfilingInfo(array->array[i].event, CL_PROFILING_COMMAND_END,
                                         sizeof(cl_ulong), &event_time, NULL);
        if (status != CL_SUCCESS) {
            ret_status = status;
            LOGE("EVENT LOGGER | Could not read end time of %s at %s:%d returned with %d\n",
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

void print_events(const event_array_t *event_array, int print_times, const char *prefix) {
    for (int i = 0; i < event_array->current_capacity; ++i) {
        if (print_times) {
            float time_ms = -1.0f;
            cl_int _ = get_event_time_ms(event_array->array[i], CL_PROFILING_COMMAND_START,
                                         CL_PROFILING_COMMAND_END, &time_ms);
            LOGI("%sEVENT LOGGER | Event %2d: %30s %8.3f ms", prefix, i,
                 event_array->array[i].description, time_ms);
        } else {
            LOGI("%sEVENT LOGGER | Event %2d: %30s", prefix, i, event_array->array[i].description);
        }
    }
}

void release_events(event_array_t *array) {
    for (int i = 0; i < array->current_capacity; i++) {
        clReleaseEvent(array->array[i].event);
    }
}

int find_event_time(const char *description, const collected_events_t *collected_events,
                    float *time_ms) {

    for (int i = 0; i < collected_events->num_events; ++i) {
        if (strcmp(description, collected_events->descriptions[i]) == 0) {
            *time_ms = collected_events->end_start_ms[i];
            return i;
        }
    }

    return -1;
}

cl_int get_event_time_ms(event_pair_t event_pair, int start_cmd, int end_cmd, float *time_ms) {
    cl_int status = CL_SUCCESS;
    cl_event event = event_pair.event;
    const char *description = event_pair.description;

    cl_ulong start_time_ns, end_time_ns;
    status = clGetEventProfilingInfo(event, start_cmd, sizeof(cl_ulong), &start_time_ns, NULL);
    if (status != CL_SUCCESS) {
        LOGE("EVENT LOGGER | Could not get profiling info of start command %#04x of event %s: %d\n",
             start_cmd, description, status);
        return status;
    }

    status = clGetEventProfilingInfo(event, end_cmd, sizeof(cl_ulong), &end_time_ns, NULL);
    if (status != CL_SUCCESS) {
        LOGE("EVENT LOGGER | Could not get profiling info of end command %#04x of event %s: %d\n",
             end_cmd, description, status);
        return status;
    }

    *time_ms = (float) (end_time_ns - start_time_ns) / 1e6f;

    return status;
}

void collect_events(const event_array_t *const event_array, collected_events_t *collected_events) {
    collected_events->num_events = event_array->current_capacity;

    for (int i = 0; i < collected_events->num_events; ++i) {
        // error value, use non-zero as some times can be close to zero and appear as zero if printed
        float time_ms = -1.0f;

        cl_int status = get_event_time_ms(event_array->array[i], CL_PROFILING_COMMAND_START,
                                          CL_PROFILING_COMMAND_END, &time_ms);

        if (status == CL_SUCCESS) {
            collected_events->end_start_ms[i] = time_ms;
        }

        collected_events->descriptions[i] = event_array->array[i].description;
    }
}

void reset_collected_events(collected_events_t *collected_events) {
    for (int i = 0; i < collected_events->num_events; ++i) {
        collected_events->descriptions[i] = NULL;
        collected_events->end_start_ms[i] = -1.0f;
    }

    collected_events->num_events = 0;
}