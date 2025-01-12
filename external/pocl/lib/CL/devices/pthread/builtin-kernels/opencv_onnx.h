#ifndef BASIC_OPENCV_ONNX_H
#define BASIC_OPENCV_ONNX_H

#include <CL/cl.h>
#include <pocl_types.h>
#include <pocl_cl.h>

#ifdef __cplusplus
extern "C" {
#endif

POCL_EXPORT
void _pocl_kernel_pocl_dnn_detection_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_segmentation_postprocess_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_segmentation_reconstruct_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_decompress_from_jpeg_rgb888_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_eval_iou_f32_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);


POCL_EXPORT
void init_onnx(cl_program program, cl_uint device_i);

POCL_EXPORT
void finish_onnx(cl_device_id device, cl_program program,
                 unsigned dev_i);

enum Task {
    DETECT,
    SEGMENT,
};

class OnnxCtx {
public:
    OnnxCtx(const std::string &onnxModelPath, Task task,
            const cv::Size &modelInputShape = {640, 480},
            const cv::Size &segmentationMaskShape = {160, 120},
            const bool &runWithCuda = true);

    void loadOnnxNetwork();

    void setRotationCwDegrees(int degrees);

    std::string modelPath;
    Task task;
    cv::Size modelShape;
    cv::Size segmentationMaskShape;
    bool cudaEnabled;
    Ort::AllocatorWithDefaultOptions ortAllocator;
    std::vector<cv::Mat> outputs;
    std::unique_ptr<Ort::Session> net;
    Ort::Env ortEnv;
    std::vector<Ort::AllocatedStringPtr> onnxInputNames;
    std::vector<Ort::AllocatedStringPtr> onnxOutputNames;

    int rotationCwDegrees = 0;
    float modelConfidenseThreshold{0.25};
    float modelScoreThreshold{0.45};
    float modelNMSThreshold{0.50};

    std::vector<std::string> classes{
            "person", "bicycle", "car",
            "motorcycle", "airplane", "bus",
            "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign",
            "parking meter", "bench", "bird",
            "cat", "dog", "horse",
            "sheep", "cow", "elephant",
            "bear", "zebra", "giraffe",
            "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove",
            "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup",
            "fork", "knife", "spoon",
            "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli",
            "carrot", "hot dog", "pizza",
            "donut", "cake", "chair",
            "couch", "potted plant", "bed",
            "dining table", "toilet", "tv",
            "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink",
            "refrigerator", "book", "clock",
            "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"};
};

void run_onnx_inference(OnnxCtx *const onnx_ctx, const unsigned char *data, int width, int height,
                        int rotate_cw_degrees, int inp_format,
                        unsigned int *output, unsigned char *out_mask);

void run_segmentation_postprocess(const OnnxCtx *const onnx_ctx, const unsigned int *detection_data,
                                  const unsigned char *segmentation_data,
                                  unsigned char *output);

void run_segmentation_reconstruct(const OnnxCtx *const onnx_ctx, const unsigned char *postprocess_data,
                                  unsigned char *output);

void run_decompress_from_jpeg_rgb888(const uint8_t *input,
                                     const uint64_t *input_size,
                                     int32_t width, int32_t height,
                                     uint8_t *output);

void eval_iou(const OnnxCtx *const onnx_ctx, const uint8_t *det_data, const uint8_t *seg_data,
              const uint8_t *ref_det_data, const uint8_t *ref_seg_data,
              int do_segment, float *iou);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_init_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_destroy_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_detection_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_segmentation_postprocess_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_segmentation_reconstruct_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void _pocl_kernel_pocl_dnn_ctx_eval_iou_f32_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z);

POCL_EXPORT
void init_onnx_ctx(cl_program program, cl_uint device_i);

POCL_EXPORT
void finish_onnx_ctx(cl_device_id device, cl_program program, unsigned dev_i);

POCL_EXPORT
void kick_onnx_awake();

#ifdef __cplusplus
}
#endif

#endif // BASIC_OPENCV_ONNX_H
