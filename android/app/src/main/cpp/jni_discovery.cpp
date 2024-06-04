#include <rename_opencl.h>
#include <jni.h>
#include "CL/cl.h"
#include <string>
#include <android/log.h>
#include <pthread.h>
#include <cassert>

#define str(x) #x

#define CHECK(x)                                                    \
    do{                                                             \
        cl_int err = (x);                                           \
        if(err != CL_SUCCESS)                                       \
        {                                                           \
            __android_log_print(ANDROID_LOG_INFO, "DISC",           \
            "CL error %d in " __FILE__ ":%d: %s\n",err,             \
            __LINE__, str(x));                                      \
            return err;                                             \
        }                                                           \
    }while(0)

cl_int add_device(char *parameter, cl_uint mode) {

    // mode => 0:add - 1:reconnect

    // Function name can also be retrieved using clGetPlatformInfo for CL_PLATFORM_EXTENSIONS
    const char funcName[] = "clAddReconnectDevicePOCL";
    // Adding or reconnecting to a device is only supported by remote driver as of now
    const char device_driver_name[] = "remote";
    cl_platform_id platform = NULL;
    cl_int (*func)(char *const, cl_uint, void *);

    CHECK(clGetPlatformIDs(1, &platform, NULL));
    assert(NULL != platform);

    func = (cl_int (*)(char *const, cl_uint, void *)) clGetExtensionFunctionAddressForPlatform(
            platform, funcName);
    assert(NULL != func);
    CHECK(func(parameter, mode, (void *) device_driver_name));

    return CL_SUCCESS;
}

extern "C"
JNIEXPORT void JNICALL
Java_org_portablecl_poclaisademo_Discovery_addDevice(JNIEnv *env, jclass clazz, jstring key,
                                                     jint mode) {
    //key - IP:port
    //mode is used to either add (mode=0) a new device or reconnect (mode=1) to a previously added device
    char *parameter = (char *) env->GetStringUTFChars(key, 0);

    add_device(parameter, (cl_uint) mode);
    env->ReleaseStringUTFChars(key, parameter);
}