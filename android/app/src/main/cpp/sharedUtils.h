//
// Created by rabijl on 21.4.2023.
//

#ifndef POCL_AISA_DEMO_SHAREDUTILS_H
#define POCL_AISA_DEMO_SHAREDUTILS_H

// todo: provide macros for other platforms

#include "platform.h"

// enable this to print timing to logs
#define PRINT_PROFILE_TIME

/**
 * when using this, don't forget to define LOGTAG before hand
 */
#define CHECK_AND_RETURN(ret, msg)                                          \
    if(ret != CL_SUCCESS) {                                                 \
        LOGE("ERROR: %s at line %d in %s returned with %d\n",               \
			 msg, __LINE__, __FILE__, ret);                                 \
        return ret;                                                         \
    }

#define  PRINT_DIFF(diff, msg)                                              \
    LOGE("%s took this long: %lu ms, %lu ns\n",                           \
        msg, ((diff) / 1000000), (diff) % 1000000);                         \

#define VAR_NAME(var) #var


#endif //POCL_AISA_DEMO_SHAREDUTILS_H
