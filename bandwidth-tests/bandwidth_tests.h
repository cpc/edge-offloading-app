//
// Created by rabijl on 22.9.2023.
//

#ifndef _BANDWIDTH_TESTS_H_
#define _BANDWIDTH_TESTS_H_

#include <CL/cl.h>

typedef struct {
  cl_command_queue queue;
  cl_kernel kernel;
  cl_mem rec_buffer;
  cl_mem rec_buffer2;
  cl_mem send_buffer;
  cl_mem send_buffer2;
  size_t global_wg;
} state_t;

int
setup_bandwidth_single_buffer (int buf_size, state_t *state);

int
run_bandwidth_single_buffer (int repeats, state_t *state);

void
destroy_bandwidth_state (state_t *state);

int
setup_bandwidth_double_buffer (int buf_size, state_t *state);

int
run_bandwidth_double_buffer (int repeats, state_t *state);


#endif //_BANDWIDTH_TESTS_H_
