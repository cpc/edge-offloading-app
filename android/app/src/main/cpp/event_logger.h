//
// Created by rabijl on 8/28/23.
// headers related to logging events for pocl image processor
//

#ifndef POCL_AISA_DEMO_EVENT_LOGGER_H
#define POCL_AISA_DEMO_EVENT_LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "rename_opencl.h"
#include <CL/cl.h>
#include <malloc.h>

#define MAX_EVENTS 64

typedef struct {
    cl_event event;
    const char *description;
} event_pair_t;

typedef struct {
    int max_capacity;
    int current_capacity;
    event_pair_t *array;
} event_array_t;

/**
 * Contains measured event times (in milliseconds) of each event in event_array_t.
 *
 * Currently holds only the end-start times of the events, but more can be added as needed.
 */
typedef struct {
    int num_events;
    const char *descriptions[MAX_EVENTS];
    float end_start_ms[MAX_EVENTS];
} collected_events_t;

/**
 * create an array of maximum size 'size'
 * @param size
 * @return
 */
event_array_t create_event_array(const int size);

event_array_t *create_event_array_pointer(const int size);

void free_event_array_pointer(event_array_t **array);

/**
 * reset the current index to 0. Doesn't clear data.
 * @param array
 */
void reset_event_array(event_array_t *array);

/**
 * free the array of event pairs.
 * @param array
 */
void free_event_array(event_array_t *array);

/**
 * append an event and description to array
 * @param array
 * @param event
 * @param description
 */
void append_to_event_array(event_array_t *array, cl_event event, const char *description);

/**
 * log event-related data to file descriptor fd.
 * @param fd file descriptor to write to
 * @param frame_index of current events
 * @param array
 * @return
 */
int log_events(int fd, int frame_index, const event_array_t *array);

/**
 * Print all event names of an array to stdout (useful for debugging)
 */
void print_events(const event_array_t *event_array, int print_times, const char *prefix);

/**
 * call clReleaseEvent on all events.
 * @param array
 */
void release_events(event_array_t *array);

/**
 * Find a position of an event in the collected events by name (-1 if not found)
 */
int find_event_time(const char *description, const collected_events_t *collected_events,
                    float *time_ms);

/**
 * Get event time in milliseconds between two profiling commands (most commonly
 * CL_PROFILING_COMMAND_START and CL_PROFILING_COMMAND_END).
 *
 * Returns the exit code of clGetEventProfilingInfo()
 */
cl_int get_event_time_ms(event_pair_t event_pair, int start_cmd, int end_cmd, float *time_ms);

/**
 * Read events in the array and collect their times (the events must have finished running)
 */
void collect_events(const event_array_t *const event_array, collected_events_t *collected_events);

/**
 * Fill collected event results with error values and set their number to 0
 */
void reset_collected_events(collected_events_t *collected_events);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_EVENT_LOGGER_H
