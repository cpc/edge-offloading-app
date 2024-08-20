
cmake_minimum_required(VERSION 3.22.1)
add_executable(test_4b test_4b.cpp
        ${APP_DIR}/opencl_utils.cpp ${APP_DIR}/opencl_utils.hpp
        ${APP_DIR}/segment_4b_compression.cpp ${APP_DIR}/segment_4b_compression.hpp
        ${APP_DIR}/sharedUtils.h ${APP_DIR}/sharedUtils.c
        ${APP_DIR}/event_logger.c ${APP_DIR}/event_logger.h
        )

target_include_directories(test_4b PUBLIC
        ${EXTERNAL_DIR}/pocl/include
        ${APP_DIR})

add_dependencies(test_4b pocl)

configure_file(data/segmentation.bin data/segmentation.bin COPYONLY)
configure_file(data/detections.bin data/detections.bin COPYONLY)

target_link_libraries(test_4b
        libpocl
        OpenCL
        ${LTTNG_UST_LDFLAGS}
        )

add_executable(TestPingThread TestPingThread.cpp
        ${APP_DIR}/opencl_utils.cpp ${APP_DIR}/opencl_utils.hpp
        ${APP_DIR}/testapps.cpp ${APP_DIR}/testapps.h
        ${APP_DIR}/PingThread.cpp ${APP_DIR}/PingThread.h
        ${APP_DIR}/sharedUtils.h ${APP_DIR}/sharedUtils.c
        ${APP_DIR}/event_logger.c ${APP_DIR}/event_logger.h)

target_include_directories(TestPingThread PUBLIC
        ${EXTERNAL_DIR}/pocl/include
        ${APP_DIR})

add_dependencies(TestPingThread pocl)

target_link_libraries(TestPingThread
        libpocl
        OpenCL
        ${LTTNG_UST_LDFLAGS})