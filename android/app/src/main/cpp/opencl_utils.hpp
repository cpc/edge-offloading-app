#include <fstream>
#include <string>
#include <vector>

#include <CL/cl.h>

// Translate OpenCL error code to string, taken from:
// https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
// https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
static const char *opencl_error_string(cl_int error) {
    switch (error) {
        // run-time and JIT compiler errors
        case 0:
            return "CL_SUCCESS";
        case -1:
            return "CL_DEVICE_NOT_FOUND";
        case -2:
            return "CL_DEVICE_NOT_AVAILABLE";
        case -3:
            return "CL_COMPILER_NOT_AVAILABLE";
        case -4:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5:
            return "CL_OUT_OF_RESOURCES";
        case -6:
            return "CL_OUT_OF_HOST_MEMORY";
        case -7:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8:
            return "CL_MEM_COPY_OVERLAP";
        case -9:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case -10:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11:
            return "CL_BUILD_PROGRAM_FAILURE";
        case -12:
            return "CL_MAP_FAILURE";
        case -13:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15:
            return "CL_COMPILE_PROGRAM_FAILURE";
        case -16:
            return "CL_LINKER_NOT_AVAILABLE";
        case -17:
            return "CL_LINK_PROGRAM_FAILURE";
        case -18:
            return "CL_DEVICE_PARTITION_FAILED";
        case -19:
            return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

            // compile-time errors
        case -30:
            return "CL_INVALID_VALUE";
        case -31:
            return "CL_INVALID_DEVICE_TYPE";
        case -32:
            return "CL_INVALID_PLATFORM";
        case -33:
            return "CL_INVALID_DEVICE";
        case -34:
            return "CL_INVALID_CONTEXT";
        case -35:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case -36:
            return "CL_INVALID_COMMAND_QUEUE";
        case -37:
            return "CL_INVALID_HOST_PTR";
        case -38:
            return "CL_INVALID_MEM_OBJECT";
        case -39:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40:
            return "CL_INVALID_IMAGE_SIZE";
        case -41:
            return "CL_INVALID_SAMPLER";
        case -42:
            return "CL_INVALID_BINARY";
        case -43:
            return "CL_INVALID_BUILD_OPTIONS";
        case -44:
            return "CL_INVALID_PROGRAM";
        case -45:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46:
            return "CL_INVALID_KERNEL_NAME";
        case -47:
            return "CL_INVALID_KERNEL_DEFINITION";
        case -48:
            return "CL_INVALID_KERNEL";
        case -49:
            return "CL_INVALID_ARG_INDEX";
        case -50:
            return "CL_INVALID_ARG_VALUE";
        case -51:
            return "CL_INVALID_ARG_SIZE";
        case -52:
            return "CL_INVALID_KERNEL_ARGS";
        case -53:
            return "CL_INVALID_WORK_DIMENSION";
        case -54:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case -55:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case -56:
            return "CL_INVALID_GLOBAL_OFFSET";
        case -57:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case -58:
            return "CL_INVALID_EVENT";
        case -59:
            return "CL_INVALID_OPERATION";
        case -60:
            return "CL_INVALID_GL_OBJECT";
        case -61:
            return "CL_INVALID_BUFFER_SIZE";
        case -62:
            return "CL_INVALID_MIP_LEVEL";
        case -63:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64:
            return "CL_INVALID_PROPERTY";
        case -65:
            return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66:
            return "CL_INVALID_COMPILER_OPTIONS";
        case -67:
            return "CL_INVALID_LINKER_OPTIONS";
        case -68:
            return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69:
            return "CL_INVALID_PIPE_SIZE";
        case -70:
            return "CL_INVALID_DEVICE_QUEUE";

            // extension errors
        case -1000:
            return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001:
            return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002:
            return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003:
            return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004:
            return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005:
            return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006:
            return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007:
            return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008:
            return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009:
            return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010:
            return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011:
            return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012:
            return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013:
            return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1057:
            return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058:
            return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059:
            return "CL_INVALID_PARTITION_NAME_EXT";
        case -1092:
            return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1093:
            return "CL_INVALID_EGL_OBJECT_KHR";
        case -1094:
            return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095:
            return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096:
            return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097:
            return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1098:
            return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099:
            return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100:
            return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101:
            return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        case -9999:
            return "NVidia - Illegal read or write to a buffer";
        default:
            return "Unknown OpenCL error";
    }
}

// Various checking helpers
static void log_if_cl_err(cl_int status, const char *msg) {
    if (status == CL_SUCCESS) {
        return;
    }

    std::string err_msg("OpenCL: CL Error ");
    err_msg.append(std::to_string(status));
    err_msg.append(" (");
    err_msg.append(opencl_error_string(status));
    err_msg.append(") '");
    err_msg.append(msg);
    err_msg.append("'");

    printf("%s\n", err_msg.c_str());
}

static void throw_if_nullptr(void *ptr, const char *msg) {
    if (ptr == nullptr) {
        std::string err_msg("OpenCL: Null pointer: '");
        err_msg.append(msg);
        err_msg.append("'");

        printf("%s\n", err_msg.c_str());
        exit(1);
    }
}

static void throw_if_zero(int x, const char *msg) {
    if (x == 0) {
        std::string err_msg("OpenCL: ");
        err_msg.append(msg);

        printf("%s\n", err_msg.c_str());
        exit(1);
    }
}

// Read contents of files
static std::vector<std::string>
read_files(const std::vector<std::string> &filenames) {
    std::vector<std::string> contents(filenames.size());

    for (u_int i = 0; i < filenames.size(); ++i) {
        std::ifstream rf(filenames[i]);
        std::string line;
        while (std::getline(rf, line)) {
            contents[i] += line;
            contents[i].push_back('\n');
        }
    }

    return contents;
}

/* Print infos about platforms */
static cl_int print_platforms_info(const cl_platform_id *platforms, u_int n) {
    cl_int status = CL_SUCCESS;

    std::string info;
    size_t str_size;
    std::vector<std::string> names = {
            "name",
            "version",
    };
    cl_platform_info platform_infos[] = {
            CL_PLATFORM_NAME,
            CL_PLATFORM_VERSION,
    };

    printf("OpenCL: %s platforms detected\n", std::to_string(n).c_str());

    for (u_int i = 0; i < n; i++) {
        printf("OpenCL: Platform %s:\n", std::to_string(i).c_str());

        for (u_int j = 0; j < names.size(); ++j) {
            status = clGetPlatformInfo(platforms[i], platform_infos[j], 0, NULL,
                                       &str_size);

            if (status != CL_SUCCESS) {
                return status;
            }

            info.resize(str_size);

            status = clGetPlatformInfo(platforms[i], platform_infos[j],
                                       str_size, (void *) info.data(), NULL);

            if (status != CL_SUCCESS) {
                return status;
            }

            printf("OpenCL: - %s: %s\n", names[j].c_str(), info.c_str());
        }
    }

    return status;
}

/* Print infos about devices */
static cl_int print_devices_info(const cl_device_id *devices, u_int n) {
    cl_int status = CL_SUCCESS;

    std::string info;
    size_t str_size;
    std::vector<std::string> names = {
            "vendor",
            "name",
            "device version",
            "OpenCL C version",
            "driver version",
            "built-in kernels",
            "profiling timer resolution (ns)",
            "max work group size",
            "extensions",
    };
    cl_device_info device_infos[] = {
            CL_DEVICE_VENDOR,
            CL_DEVICE_NAME,
            CL_DEVICE_VERSION,
            CL_DEVICE_OPENCL_C_VERSION,
            CL_DRIVER_VERSION,
            CL_DEVICE_BUILT_IN_KERNELS,
            CL_DEVICE_PROFILING_TIMER_RESOLUTION,
            CL_DEVICE_MAX_WORK_GROUP_SIZE,
            CL_DEVICE_EXTENSIONS,
    };

    printf("OpenCL: %d devices detected\n", n);

    for (u_int i = 0; i < n; i++) {
        printf("OpenCL: Device %d (%ld):\n", i, (size_t) (devices[i]));

        for (u_int j = 0; j < names.size(); ++j) {
            status = clGetDeviceInfo(devices[i], device_infos[j], 0, NULL,
                                     &str_size);

            if (status != CL_SUCCESS) {
                return status;
            }

            if (device_infos[j] == CL_DEVICE_PROFILING_TIMER_RESOLUTION ||
                device_infos[j] == CL_DEVICE_MAX_WORK_GROUP_SIZE) {
                size_t data;
                status = clGetDeviceInfo(devices[i], device_infos[j],
                                         sizeof(size_t), (void *) &data, NULL);
                info = std::to_string(data);
            } else {
                info.resize(str_size);
                status = clGetDeviceInfo(devices[i], device_infos[j], str_size,
                                         (void *) info.data(), NULL);
            }

            if (status != CL_SUCCESS) {
                return status;
            }

            printf("OpenCL: - %s: %s\n", names[j].c_str(), info.c_str());
        }
    }

    return status;
}

/* Print build error */
static cl_int print_build_error_info(const cl_program &program,
                                     const std::vector<cl_device_id> &devices) {
    cl_int status;
    cl_build_status build_status;

    for (u_int i = 0; i < devices.size(); ++i) {
        status =
                clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_STATUS,
                                      sizeof(cl_build_status), &build_status, NULL);
        if (status != CL_SUCCESS) {
            return status;
        }

        // // don't print if build was successful
        // if (build_status == CL_SUCCESS) {
        //     continue;
        // }

        std::string info;
        size_t str_size;

        status = clGetProgramBuildInfo(
                program, devices[i], CL_PROGRAM_BUILD_LOG, 0, NULL, &str_size);
        if (status != CL_SUCCESS) {
            return status;
        }

        info.resize(str_size);

        status =
                clGetProgramBuildInfo(program, devices[i], CL_PROGRAM_BUILD_LOG,
                                      str_size, (void *) info.data(), NULL);
        if (status != CL_SUCCESS) {
            return status;
        }

        printf("Build error:\n%s", info.c_str());
    }

    return CL_SUCCESS;
}

void *
read_bin_file(char const *file_location, size_t *const file_size);

int
write_bin_file(char const *const file_location, void const *const content, size_t const file_size);

void print_program_build_log(cl_program program, cl_device_id device);