#include <stdio.h>
#include "bandwidth_tests.h"

#define RUN_SINGLE
#define RUN_DOUBLE

int main ()
{
  int buff_size;
  buff_size = 1000000;
//  buff_size = 115200; //size of one yuv image
//  buff_size = 100000;
//  buff_size = 10000;
//  buff_size = 1000;

  int repeats = 100;

  state_t  state;

#ifdef RUN_SINGLE

  setup_bandwidth_single_buffer (buff_size, &state);

  run_bandwidth_single_buffer (repeats, &state);

  destroy_bandwidth_state (&state);

#endif

#ifdef RUN_DOUBLE

  setup_bandwidth_double_buffer (buff_size, &state);

  run_bandwidth_double_buffer(repeats, &state);

  destroy_bandwidth_state (&state);
#endif

  return 0;
}
