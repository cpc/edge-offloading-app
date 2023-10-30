//
// Created by rabijl on 8/28/23.
// headers related to logging events for pocl image processor
//

#ifndef POCL_AISA_DEMO_EVENT_LOGGER_H
#define POCL_AISA_DEMO_EVENT_LOGGER_H

#ifdef __cplusplus
extern "C" {
#endif

#if __ANDROID__
// required for proxy device http://portablecl.org/docs/html/proxy.html
#include <rename_opencl.h>

#endif

#include <CL/cl.h>
#include <malloc.h>

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
 * create an array of maximum size 'size'
 * @param size
 * @return
 */
event_array_t
create_event_array(const int size);

/**
 * reset the current index to 0. Doesn't clear data.
 * @param array
 */
void
reset_event_array(event_array_t *array);

/**
 * free the array of event pairs.
 * @param array
 */
void
free_event_array(event_array_t *array);

/**
 * append an event and description to array
 * @param array
 * @param event
 * @param description
 */
void
append_to_event_array(event_array_t *array, cl_event event, const char *description);

/**
 * log event-related data to file descriptor fd.
 * @param fd file descriptor to write to
 * @param frame_index of current events
 * @param array
 * @return
 */
int
print_events(int fd, int frame_index, const event_array_t *array);

/**
 * call clReleaseEvent on all events.
 * @param array
 */
void
release_events(event_array_t *array);

#ifdef __cplusplus
}
#endif

#endif //POCL_AISA_DEMO_EVENT_LOGGER_H
