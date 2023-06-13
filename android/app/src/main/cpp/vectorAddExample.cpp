//
// Created by rabijl on 29.3.2023.
//
#include "vectorAddExample.h"
#include <CL/cl.h>
#include <android/log.h>
#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


#define LOCAL_SIZE  64


#define CHECK_AND_RETURN(ret, msg)                                          \
    if(ret != CL_SUCCESS) {                                                 \
        __android_log_print(ANDROID_LOG_ERROR, "openCL vector add",         \
				"ERROR: %s at line %d in %s returned with %d\n",            \
					msg, __LINE__, __FILE__, ret);                          \
        return ret;                                                         \
    }


static const char *vector_add_str="											\
		__kernel void vec_add(int N, __global float *A,						\
								__global float *B, __global float *C)		\
		{																	\
			int id = get_global_id(0);										\
																			\
			if(id < N) {													\
				C[id] = A[id] + B[id];										\
			}																\
		}																	\
		";


static cl_context clContext = NULL;
static cl_command_queue clCommandQueue = NULL;
static cl_program clProgram = NULL;
static cl_kernel clKernel = NULL;

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_initCL(JNIEnv *env, jobject thiz)
{

    __android_log_print(ANDROID_LOG_ERROR, "openCL vector add", "If you see this, logging from native code is working.");

    cl_platform_id clPlatform;
    cl_device_id clDevice;
    cl_int	status;

    status = clGetPlatformIDs(1, &clPlatform, NULL);
    CHECK_AND_RETURN(status, "getting platform id failed");

    status = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_DEFAULT, 1, &clDevice, NULL);
    CHECK_AND_RETURN(status, "getting device id failed");

    // some info
    char result_array[256];
    clGetDeviceInfo(clDevice, CL_DEVICE_NAME, 256* sizeof (char), result_array,NULL);
    __android_log_print(ANDROID_LOG_INFO, "openCL vector add", "CL_DEVICE_NAME: %s", result_array);

    clGetDeviceInfo(clDevice, CL_DEVICE_VERSION, 256* sizeof (char), result_array,NULL);
    __android_log_print(ANDROID_LOG_INFO, "openCL vector add", "CL_DEVICE_VERSION: %s", result_array);

    clGetDeviceInfo(clDevice, CL_DRIVER_VERSION, 256* sizeof (char), result_array,NULL);
    __android_log_print(ANDROID_LOG_INFO, "openCL vector add", "CL_DRIVER_VERSION: %s", result_array);

    clGetDeviceInfo(clDevice, CL_DEVICE_NAME, 256* sizeof (char), result_array,NULL);
    __android_log_print(ANDROID_LOG_INFO, "openCL vector add", "CL_DEVICE_NAME: %s", result_array);


    cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) clPlatform,
                                    0 };

    clContext = clCreateContext(cps, 1, &clDevice, NULL, NULL, &status);
    CHECK_AND_RETURN(status, "creating context failed");

    clCommandQueue = clCreateCommandQueue(clContext, clDevice, 0, &status);
    CHECK_AND_RETURN(status, "creating command queue failed");

    size_t strSize = strlen(vector_add_str);
    clProgram = clCreateProgramWithSource(clContext, 1, &vector_add_str, &strSize, &status);
    CHECK_AND_RETURN(status, "creating program failed");

    status = clBuildProgram(clProgram, 1, &clDevice, NULL, NULL, NULL);
    CHECK_AND_RETURN(status, "build program failed");

    clKernel = clCreateKernel(clProgram, "vec_add", &status);
    CHECK_AND_RETURN(status, "creating kernel failed");



    return 0;
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_destroyCL(JNIEnv *env, jobject thiz)
{
    if(clKernel)		clReleaseKernel(clKernel);
    if(clProgram)		clReleaseProgram(clProgram);
    if(clCommandQueue) 	clReleaseCommandQueue(clCommandQueue);
    if(clContext)	    clReleaseContext(clContext);

    return 0;
}

JNIEXPORT jint JNICALL
Java_org_portablecl_poclaisademo_MainActivity_vectorAddCL(JNIEnv *env, jobject thiz, jint n,
                                                          jfloatArray a, jfloatArray b,
                                                          jfloatArray c)
{
    cl_int	status;
    int byteSize = n * sizeof(float);

    // Get pointers to array from jni wrapped floatArray
    jfloat* A = env->GetFloatArrayElements(a, 0);
    jfloat* B = env->GetFloatArrayElements(b, 0);
    jfloat* C = env->GetFloatArrayElements(c, 0);


    cl_mem A_obj = clCreateBuffer(clContext, (CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR),
                                  byteSize, A, &status);
    CHECK_AND_RETURN(status, "create buffer A failed");

    cl_mem B_obj = clCreateBuffer(clContext, (CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR),
                                  byteSize, B, &status);
    CHECK_AND_RETURN(status, "create buffer B failed");

    cl_mem C_obj = clCreateBuffer(clContext, (CL_MEM_WRITE_ONLY),
                                  byteSize, NULL, &status);
    CHECK_AND_RETURN(status, "create buffer C failed");

    status = clSetKernelArg(clKernel, 0, sizeof(cl_int), (void *)&n);
    status |= clSetKernelArg(clKernel, 1, sizeof(cl_mem), (void *)&A_obj);
    status |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), (void *)&B_obj);
    status |= clSetKernelArg(clKernel, 3, sizeof(cl_mem), (void *)&C_obj);
    CHECK_AND_RETURN(status, "clSetKernelArg failed");

    size_t localSize = LOCAL_SIZE;
    size_t wgs = (n + localSize - 1) / localSize;
    size_t globalSize = wgs * localSize;

    status = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 1, NULL,
                                    &globalSize, &localSize, 0, NULL, NULL);
    CHECK_AND_RETURN(status, "clEnqueueNDRange failed");

    status = clEnqueueReadBuffer(clCommandQueue,C_obj,CL_TRUE,0,byteSize,C,0,NULL,NULL);
    CHECK_AND_RETURN(status, "clEnqueueReadBuffer failed");

    status = clFinish(clCommandQueue);
    CHECK_AND_RETURN(status, "clFinish failed");

    env->ReleaseFloatArrayElements(a, A, 0);
    env->ReleaseFloatArrayElements(b, B, 0);
    env->ReleaseFloatArrayElements(c, C, 0);

    return 0;
}



JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_MainActivity_setenv(JNIEnv *env, jobject thiz, jstring key,
jstring value)
{
    setenv((char*) env->GetStringUTFChars(key, 0),
           (char*) env->GetStringUTFChars(value, 0), 1);
}

#ifdef __cplusplus
}
#endif