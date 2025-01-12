#include "pocl_cl.h"
#include "devices/devices.h"

CL_API_ENTRY cl_int CL_API_CALL
POname(clAddDevicePOCL) (char* parameters, cl_bool mode, void *driver_id)
CL_API_SUFFIX__VERSION_1_2
{
    int errcode;
    
    POCL_RETURN_ERROR_ON ((driver_id == NULL), CL_INVALID_VALUE,
                        "Driver id corresponding to the device cannot be null.\n");

    // error CL_INVALID_DEVICE_TYPE, CL_DEVICE_NOT_FOUND should never happen, however check is added to return CL_DEVICE_NOT_AVAILABLE if it occurs
    errcode = pocl_add_reconnect_device(parameters, mode, driver_id);
    if (errcode == CL_INVALID_DEVICE_TYPE || errcode == CL_DEVICE_NOT_FOUND)
        errcode = CL_DEVICE_NOT_AVAILABLE;

    return errcode;
}

POsym (clAddDevicePOCL)
