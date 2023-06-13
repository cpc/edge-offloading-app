//
// Created by rabijl on 21.4.2023.
//

#ifndef POCL_AISA_DEMO_SHAREDUTILS_H
#define POCL_AISA_DEMO_SHAREDUTILS_H


/**
 * when using this, don't forget to define LOGTAG before hand
 */
#define CHECK_AND_RETURN(ret, msg)                                          \
    if(ret != CL_SUCCESS) {                                                 \
        __android_log_print(ANDROID_LOG_ERROR, LOGTAG,         \
				"ERROR: %s at line %d in %s returned with %d\n",            \
					msg, __LINE__, __FILE__, ret);                          \
        return ret;                                                         \
    }

#endif //POCL_AISA_DEMO_SHAREDUTILS_H
