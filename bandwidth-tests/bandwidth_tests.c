//
// Created by rabijl on 22.9.2023.
//

#define CL_TARGET_OPENCL_VERSION 300

#include <string.h>
#include <assert.h>
#include "sharedUtils.h"
#include "bandwidth_tests.h"

#define  NS_IN_MS 1000000

#define MAX(a,b) \
        (a > b ? a : b)

#define MIN(a,b) \
        (a < b ? a : b)

const char *kernel_source_double = " __kernel \n"
                            "void copy(__global int *input,__global int *input2 , __global int *output, __global int *output2)\n"
                            "{\n"
                            "    int index = get_global_id(0);\n"
                            "    output[index] = input[index];\n"
                            "    output2[index] = input2[index];\n"
                            "}";

const char *kernel_source_single = " __kernel \n"
                            "void copy(__global int *input, __global int *output)\n"
                            "{\n"
                            "    int index = get_global_id(0);\n"
                            "    output[index] = input[index];\n"
                            "}";



//state_t state = {NULL, NULL, NULL, NULL, 0};

void
init_state_to_null(state_t *state){
  state->queue = NULL;
  state-> kernel = NULL;
  state->rec_buffer = NULL;
  state->rec_buffer2 = NULL;
  state->send_buffer = NULL;
  state->send_buffer2 = NULL;
  state->global_wg = 0;
}


int
setup_opencl(cl_context *context, cl_program * program, const char ** source, state_t * state ) {

  int status;
  cl_uint num_devs;
  cl_device_id device;
    cl_platform_id platform;
  char result_array[256];

  status = clGetPlatformIDs (1, &platform, &num_devs);
  CHECK_AND_RETURN(status, "could not query for platforms");
  clGetPlatformInfo (platform, CL_PLATFORM_NAME, 256, result_array, NULL);
  LOGI("CL_DEVICE_NAME: %s\n", result_array);

  status = clGetDeviceIDs (platform, CL_DEVICE_TYPE_ALL, 1, &device, &num_devs);
  CHECK_AND_RETURN(status, "could not get device");

  clGetDeviceInfo (device, CL_DEVICE_NAME, 256 * sizeof (char), result_array, NULL);
  LOGI("CL_DEVICE_NAME: %s\n", result_array);

  cl_context_properties cps[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
                                 0};
  *context = clCreateContext (cps, 1, &device, NULL, NULL, &status);
  CHECK_AND_RETURN(status, "creating context failed");

  cl_command_queue_properties cq_properties = CL_QUEUE_PROFILING_ENABLE;
  state->queue = clCreateCommandQueue (*context, device, cq_properties, &status);
  CHECK_AND_RETURN(status, "creating command queue failed");

  size_t len = strlen (*source);
  *program = clCreateProgramWithSource (*context, 1, source, &len, &status);
  CHECK_AND_RETURN(status, "creating program failed");

  status = clBuildProgram (*program, 1, &device, NULL, NULL, NULL);
  CHECK_AND_RETURN(status, "building program failed");

  clReleaseDevice (device);
}


int
setup_bandwidth_single_buffer (int buf_size, state_t *state) {
  init_state_to_null (state);

  int status;

  cl_context context;
  cl_program program;

  setup_opencl (&context, &program, &kernel_source_single, state);

  state->rec_buffer = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create rec buffer");

  state->send_buffer = clCreateBuffer (context, CL_MEM_READ_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create send buffer");

  state->kernel = clCreateKernel (program, "copy", &status);
  CHECK_AND_RETURN(status, "could not create kernel");

  status = clSetKernelArg (state->kernel, 0, sizeof (cl_mem), &state->send_buffer);
  status |= clSetKernelArg (state->kernel, 1, sizeof (cl_mem), &state->rec_buffer);
  CHECK_AND_RETURN(status, "could not set kernel args");

  state->global_wg = buf_size;

  clReleaseContext (context);
  clReleaseProgram (program);

  return 0;
}

int
run_bandwidth_single_buffer (int repeats, state_t *state) {
  assert(repeats > 0);

  int status;
  cl_event write_event, copy_event, read_event;
  cl_ulong event_start_time, event_stop_time, avg, total_size;
  cl_ulong sum_read = 0, sum_write = 0;
  cl_int *input_data = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);
  cl_int *output_data = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);


  for (int i = 0; i < state->global_wg; i++)
    {
      input_data[i] = i;
    }

  for (int i = 0; i < repeats; i++)
    {
      status = clEnqueueWriteBuffer (state->queue, state->send_buffer, CL_FALSE, 0,
                                     sizeof (cl_int) * state->global_wg, input_data, 0, NULL, &write_event);
      CHECK_AND_RETURN(status, "could not enqueue write buffer");

      status = clEnqueueNDRangeKernel (state->queue, state->kernel, 1, NULL,
                                       &state->global_wg, NULL, 1, &write_event, &copy_event);
      CHECK_AND_RETURN(status, "could not enqueue kernel");

      status = clEnqueueReadBuffer (state->queue, state->rec_buffer, CL_FALSE, 0,
                                    sizeof (cl_int) * state->global_wg, output_data, 1, &copy_event, &read_event);
      CHECK_AND_RETURN(status, "could not enqueue read buffer")

      clFinish (state->queue);

      status = clGetEventProfilingInfo (write_event, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time, NULL);
      status |= clGetEventProfilingInfo (write_event, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time, NULL);
      CHECK_AND_RETURN(status, "could not read write event times");
      sum_write += event_stop_time - event_start_time;

      status = clGetEventProfilingInfo (read_event, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time, NULL);
      status |= clGetEventProfilingInfo (read_event, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time, NULL);
      CHECK_AND_RETURN(status, "could not read read event time");
      sum_read += event_stop_time - event_start_time;

      // todo: possibly check that the buffers are the same

      clReleaseEvent (write_event);
      clReleaseEvent (read_event);
      clReleaseEvent (copy_event);
    }

  total_size = state->global_wg * sizeof (cl_int) * 8;

  LOGI("****** SUMMARY SINGLE BUFFER ****** \n");
  LOGI ("runs: %d\tdata %lu b\n", repeats, total_size);
  LOGI ("average write: \n");
  avg = sum_write / repeats;
  LOGI ("%lu ms,\t%.6lu ns\n", avg / NS_IN_MS, avg % NS_IN_MS);
  // avg is in ns and we are looking at Mb so 10**9/10**6 = 1000
  LOGI ("bandwidth: %.3f Mb/s\n", (total_size * 1000.0f) / avg);
  LOGI ("average read: \n");
  avg = sum_read / repeats;
  LOGI ("%lu ms,\t%.6lu ns\n", avg / NS_IN_MS, avg % NS_IN_MS);
  LOGI ("bandwidth: %.3f Mb/s\n", (total_size * 1000.0f) / avg);

  free (input_data);
  free (output_data);


  return 0;
}

/**
 * setup everything needed to run
 * @param buf_size
 * @return
 */
int
setup_bandwidth_double_buffer (int buf_size, state_t *state)
{

  init_state_to_null (state);
  
  int status;

  cl_context context;
  cl_program program;

  setup_opencl (&context, &program, &kernel_source_double, state);

  state->rec_buffer = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create rec buffer");

  state->rec_buffer2 = clCreateBuffer (context, CL_MEM_WRITE_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create rec buffer");

  state->send_buffer = clCreateBuffer (context, CL_MEM_READ_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create send buffer");

  state->send_buffer2 = clCreateBuffer (context, CL_MEM_READ_ONLY, sizeof (cl_int) * buf_size, NULL, &status);
  CHECK_AND_RETURN(status, "could not create send buffer");

  state->kernel = clCreateKernel (program, "copy", &status);
  CHECK_AND_RETURN(status, "could not create kernel");

  status = clSetKernelArg (state->kernel, 0, sizeof (cl_mem), &state->send_buffer);
  status = clSetKernelArg (state->kernel, 1, sizeof (cl_mem), &state->send_buffer2);
  status |= clSetKernelArg (state->kernel, 2, sizeof (cl_mem), &state->rec_buffer);
  status |= clSetKernelArg (state->kernel, 3, sizeof (cl_mem), &state->rec_buffer2);
  CHECK_AND_RETURN(status, "could not set kernel args");

  state->global_wg = buf_size;

  clReleaseContext (context);
  clReleaseProgram (program);

  return 0;
}

int
run_bandwidth_double_buffer (int repeats, state_t *state)
{

  assert(repeats > 0);

  int status;
  cl_event write_event, copy_event, read_event;
  cl_event write_event2, read_event2;
  cl_event read_events[2];
  cl_ulong event_start_time, event_stop_time, avg, total_size;
  cl_ulong event_start_time2, event_stop_time2;
  cl_ulong sum_read = 0, sum_write = 0;
  cl_int *input_data = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);
  cl_int *output_data = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);
  cl_int *input_data2 = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);
  cl_int *output_data2 = (cl_int *) malloc (sizeof (cl_int) * state->global_wg);

  for (int i = 0; i < state->global_wg; i++)
    {
      input_data[i] = i;
      input_data2[i] = i;
    }

  for (int i = 0; i < repeats; i++)
    {
      status = clEnqueueWriteBuffer (state->queue, state->send_buffer, CL_FALSE, 0,
                                     sizeof (cl_int) * state->global_wg, input_data, 0, NULL, &write_event);
      CHECK_AND_RETURN(status, "could not enqueue write buffer");

      status = clEnqueueWriteBuffer (state->queue, state->send_buffer2, CL_FALSE, 0,
                                     sizeof (cl_int) * state->global_wg, input_data2, 0, NULL, &write_event2);
      CHECK_AND_RETURN(status, "could not enqueue write buffer");

      read_events[0] = write_event;
      read_events[1] = write_event2;

      status = clEnqueueNDRangeKernel (state->queue, state->kernel, 1, NULL,
                                       &state->global_wg, NULL, 2, read_events, &copy_event);
      CHECK_AND_RETURN(status, "could not enqueue kernel");

      status = clEnqueueReadBuffer (state->queue, state->rec_buffer, CL_FALSE, 0,
                                    sizeof (cl_int) * state->global_wg, output_data, 1, &copy_event, &read_event);
      CHECK_AND_RETURN(status, "could not enqueue read buffer")

      status = clEnqueueReadBuffer (state->queue, state->rec_buffer2, CL_FALSE, 0,
                                    sizeof (cl_int) * state->global_wg, output_data2, 1, &copy_event, &read_event2);
      CHECK_AND_RETURN(status, "could not enqueue read buffer")

      clFinish (state->queue);

      status = clGetEventProfilingInfo (write_event, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time, NULL);
      status |= clGetEventProfilingInfo (write_event, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time, NULL);

      status = clGetEventProfilingInfo (write_event2, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time2, NULL);
      status |= clGetEventProfilingInfo (write_event2, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time2, NULL);

      CHECK_AND_RETURN(status, "could not read write event times");
      sum_write += MAX(event_stop_time, event_stop_time2) - MIN(event_start_time2, event_start_time);

      status = clGetEventProfilingInfo (read_event, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time, NULL);
      status |= clGetEventProfilingInfo (read_event, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time, NULL);
      CHECK_AND_RETURN(status, "could not read read event time");

      status = clGetEventProfilingInfo (write_event2, CL_PROFILING_COMMAND_START,
                                        sizeof (cl_ulong),
                                        &event_start_time2, NULL);
      status |= clGetEventProfilingInfo (write_event2, CL_PROFILING_COMMAND_COMPLETE,
                                         sizeof (cl_ulong),
                                         &event_stop_time2, NULL);
      CHECK_AND_RETURN(status, "could not read read event time");

      sum_read += MAX(event_stop_time, event_stop_time2) - MIN(event_start_time2, event_start_time);

      // todo: possibly check that the buffers are the same

      clReleaseEvent (write_event);
      clReleaseEvent (read_event);
      clReleaseEvent (copy_event);
      clReleaseEvent (write_event2);
      clReleaseEvent (read_event2);
    }

  total_size = state->global_wg * sizeof (cl_int) * 8 *2;

  LOGI("****** SUMMARY DOUBLE BUFFER ****** \n");
  LOGI ("runs: %d\tdata %lu b\n", repeats, total_size);
  LOGI ("average write: \n");
  avg = sum_write / repeats;
  LOGI ("%lu ms,\t%.6lu ns\n", avg / NS_IN_MS, avg % NS_IN_MS);
  // avg is in ns and we are looking at Mb so 10**9/10**6 = 1000
  LOGI ("bandwidth: %.3f Mb/s\n", (total_size * 1000.0f) / avg);
  LOGI ("average read: \n");
  avg = sum_read / repeats;
  LOGI ("%lu ms,\t%.6lu ns\n", avg / NS_IN_MS, avg % NS_IN_MS);
  LOGI ("bandwidth: %.3f Mb/s\n", (total_size * 1000.0f) / avg);

  free (input_data);
  free (output_data);
  free (input_data2);
  free (output_data2);

  return 0;
}

void
destroy_bandwidth_state(state_t *state)
{

  if (NULL != state->kernel)
    {
      clReleaseKernel (state->kernel);
    }

  if (NULL != state->queue)
    {
      clReleaseCommandQueue (state->queue);
    }

  if (NULL != state->rec_buffer)
    {
      clReleaseMemObject (state->rec_buffer);
    }

  if (NULL != state->send_buffer)
    {
      clReleaseMemObject (state->send_buffer);
    }
  if (NULL != state->rec_buffer2)
    {
      clReleaseMemObject (state->rec_buffer2);
    }

  if (NULL != state->send_buffer2)
    {
      clReleaseMemObject (state->send_buffer2);
    }

}