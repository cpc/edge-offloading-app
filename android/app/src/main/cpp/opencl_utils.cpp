//
// Created by rabijl on 9.7.2024.
//

#include "opencl_utils.hpp"

/**
 * the the binary contents of a file
 * @param file_location
 * @param file_size out: size of the read file
 * @return pointer to file contents
 */
void *
read_bin_file(char const *file_location, size_t *const file_size) {

    FILE *file;
    void *ret;

    file = fopen(file_location, "rb");
    if (NULL == file) {
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    ret = malloc(*file_size);
    if (ret == NULL) {
        fclose(file);
        return NULL;
    }

    fseek(file, 0, SEEK_SET);
    fread(ret, *file_size, 1, file);
    fclose(file);

    return ret;
}

/**
 * write the contents of a buffer to a file
 * @param file_location
 * @param content pointer to the contents to be written
 * @param file_size size of the contents
 * @return 0 on success, otherwise less than 0
 */
int
write_bin_file(char const *const file_location, void const *const content, size_t const file_size) {

    FILE *file;

    file = fopen(file_location, "wb");
    if (NULL == file) {
        return -1;
    }

    size_t data_written = fwrite(content, 1, file_size, file);
    fclose(file);
    if (data_written != file_size) {
        return -2;
    }
    return 0;

}

/**
 * print the build log when building a kernel
 * @param program program building kernel
 * @param device specific device that the kernel was built for
 */
void print_program_build_log(cl_program program, cl_device_id device) {

    // Determine the size of the log
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    // Allocate memory for the log
    char *log = (char *) malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("program build log: %s\n", log);
    free(log);

}
