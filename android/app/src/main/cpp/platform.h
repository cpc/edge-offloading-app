#ifndef PLATFORM_H
#define PLATFORM_H

/*
 * Platform-specific definitions
 */

#ifndef LOGTAG
#define LOGTAG "poclimageprocessor"
#endif

#if __ANDROID__

#include <android/log.h>

#define LOGI(...) \
    ((void)__android_log_print(ANDROID_LOG_INFO, LOGTAG, __VA_ARGS__))
#define LOGW(...) \
    ((void)__android_log_print(ANDROID_LOG_WARN, LOGTAG, __VA_ARGS__))
#define LOGE(...) \
    ((void)__android_log_print(ANDROID_LOG_ERROR, LOGTAG, __VA_ARGS__))

#else // __ANDROID__

#include <stdio.h>
#define LOGI(...) printf(__VA_ARGS__)
#define LOGW(...) printf("WARNING: " __VA_ARGS__)
#define LOGE(...) printf("ERROR: " __VA_ARGS__)

#endif // __ANDROID__

#endif // PLATFORM_H
