#include <limits.h>
#include <optional>
#include <random>
#include <string>
#include <sys/time.h>
#include <vector>
#include <cstdio>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "config.h"

#ifdef USE_PREBUILT_ONNX_BINARY
#include <onnxruntime_cxx_api.h>
#else
#include <onnxruntime/onnxruntime_cxx_api.h>
#endif

#ifdef __ANDROID__
#include <nnapi_provider_factory.h>
#else
// We can try TensorRT later; Seems to run faster but has some errors
// #include <tensorrt_provider_factory.h>
#endif

#include <pocl_cl.h>
#include <pocl_debug.h>

#ifdef TRACY_ENABLE
#include <Tracy.hpp>
#endif

#include "opencv_onnx.h"

#define NUM_CLASSES 81
#define NO_CLASS_ID (NUM_CLASSES - 1)  // last ID signalizes no detection

static const int SEGMENTATION_COLORS[256] = {
    -1651865, -6634562, -5921894, -9968734, -1277957, -2838283,
    -9013359, -9634954, -470042, -8997255, -4620585, -2953862,
    -3811878, -8603498, -2455171, -5325920, -6757258, -8214427,
    -5903423, -4680978, -4146958, -602947, -5396049, -9898511,
    -8346466, -2122577, -2304523, -4667802, -222837, -4983945,
    -234790, -8865559, -4660525, -3744578, -8720427, -9778035,
    -680538, -7942224, -7162754, -2986121, -8795194, -2772629,
    -4820488, -9401960, -3443339, -1781041, -4494168, -3167240,
    -7629631, -6685500, -6901785, -2968136, -3953703, -4545430,
    -6558846, -2631687, -5011272, -4983118, -9804322, -2593374,
    -8473686, -4006938, -7801488, -7161859, -4854121, -5654350,
    -817410, -8013957, -9252928, -2240041, -3625560, -6381719,
    -4674608, -5704237, -8466309, -1788449, -7283030, -5781889,
    -4207444, -8225948, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0
};

struct Detection {
    int class_id{0};
    std::string className;
    float confidence{0.0};
    cv::Scalar color;
    cv::Rect box;
};

OnnxCtx::OnnxCtx(const std::string &onnxModelPath, Task task,
                 const cv::Size &modelInputShape,
                 const cv::Size &segmentationMaskShape,
                 const bool &runWithCuda) {
    this->modelPath = onnxModelPath;
    this->task = task;
    this->modelShape = modelInputShape;
    this->segmentationMaskShape = segmentationMaskShape;
    this->cudaEnabled = runWithCuda;
    this->ortEnv = Ort::Env{ORT_LOGGING_LEVEL_ERROR, "Default"};

    this->loadOnnxNetwork();
}

void OnnxCtx::setRotationCwDegrees(int degrees) {
    this->rotationCwDegrees = degrees;
}

// Smuggling-related code for the basic device
// #ifdef __ANDROID__
// // Defined and initialized from JNI code since it has to go through
// // AAssetManager
// extern const char *pocl_onnx_blob;
// extern uint64_t pocl_onnx_blob_size;
// #endif

void OnnxCtx::loadOnnxNetwork() {

#ifdef TRACY_ENABLE
    ZoneScoped;
#endif

    Ort::SessionOptions so;

#ifdef __ANDROID__
    uint32_t nnapi_flags = 0;
    Ort::ThrowOnError(
        OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
    //this->net = std::make_unique<Ort::Session>(ortEnv, pocl_onnx_blob,
    //                                           pocl_onnx_blob_size, so);
#else
    if (this->cudaEnabled) {

        POCL_MSG_PRINT_INFO("DNN: Running on CUDA\n");

        // We can later try TensorRT provider
        // const OrtTensorRTProviderOptions opts{};
        // so.AppendExecutionProvider_TensorRT(opts);
        const OrtCUDAProviderOptions opts;
        so.AppendExecutionProvider_CUDA(opts);
    } else {
        POCL_MSG_PRINT_INFO("DNN: Running on CPU\n");
    }
    //this->net =
    //    std::make_unique<Ort::Session>(ortEnv, this->modelPath.c_str(), so);
#endif
    this->net =
        std::make_unique<Ort::Session>(ortEnv, this->modelPath.c_str(), so);

    // TODO: check that this is needed
    this->onnxInputNames.reserve(this->net->GetInputCount());
    for (size_t i = 0; i < this->onnxInputNames.size(); ++i) {
        this->onnxInputNames.push_back(
            this->net->GetInputNameAllocated(i, this->ortAllocator));
    }
    this->onnxOutputNames.reserve(this->net->GetOutputCount());
    for (size_t i = 0; i < this->onnxOutputNames.size(); ++i) {
        this->onnxOutputNames.push_back(
            this->net->GetOutputNameAllocated(i, this->ortAllocator));
    }
}

namespace {
OnnxCtx *global_onnx_ctx = nullptr;
}

// Check pthread_utils.c setup_kernel_arg_array() to figure out how to get the
// args.
void _pocl_kernel_pocl_dnn_detection_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z
) {
    void **arguments = *(void ***)(args);
    void **arguments2 = (void **)(args);

    int nargs = 0;
    const unsigned char *data = (const unsigned char*)(arguments[nargs++]);
    int width = *(int*)(arguments2[nargs++]);
    int height = *(int*)(arguments2[nargs++]);
    int rotate_cw_degrees = *(int*)(arguments2[nargs++]);
    int inp_format = *(int*)(arguments2[nargs++]);
    unsigned int *output = (unsigned int *)(arguments[nargs++]);
    unsigned char *out_mask  = (unsigned char *)(arguments[nargs++]);

    run_onnx_inference(global_onnx_ctx, data, width, height, rotate_cw_degrees, inp_format,
                       output, out_mask);
}

void _pocl_kernel_pocl_dnn_segmentation_postprocess_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z
) {
    void **arguments = *(void ***)(args);

    int nargs = 0;
    const unsigned int *detection_data = (const unsigned int *)(arguments[nargs++]);
    const unsigned char *segmentation_data = (const unsigned char *)(arguments[nargs++]);
    unsigned char *output = (unsigned char*)(arguments[nargs++]);

    run_segmentation_postprocess(global_onnx_ctx, detection_data, segmentation_data, output);
}

void _pocl_kernel_pocl_dnn_segmentation_reconstruct_u8_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z
) {
    void **arguments = *(void ***)(args);

    int nargs = 0;
    const unsigned char *postprocess_data = (const unsigned char *)(arguments[nargs++]);
    unsigned char *output = (unsigned char*)(arguments[nargs++]);

    run_segmentation_reconstruct(global_onnx_ctx, postprocess_data, output);
}

void _pocl_kernel_pocl_decompress_from_jpeg_rgb888_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z
) {
    void **arguments = *(void ***)(args);
    void **arguments2 = (void **)(args);

    int nargs = 0;
    const uint8_t *input = (const uint8_t *)(arguments[nargs++]);
    const uint64_t *input_size = (const uint64_t *)(arguments[nargs++]);
    int32_t width = *(int32_t *)(arguments2[nargs++]);
    int32_t height = *(int32_t *)(arguments2[nargs++]);
    uint8_t *output = (uint8_t *)(arguments[nargs++]);

    run_decompress_from_jpeg_rgb888(input, input_size, width, height, output);
}

void _pocl_kernel_pocl_dnn_eval_iou_f32_workgroup(
    cl_uchar *args, cl_uchar *context,
    ulong group_x, ulong group_y,
    ulong group_z
) {
    void **arguments = *(void ***)(args);
    void **arguments2 = (void **)(args);

    int nargs = 0;
    const uint8_t *det_data = (const uint8_t *)(arguments[nargs++]);
    const uint8_t *seg_data = (const uint8_t *)(arguments[nargs++]);
    const uint8_t *ref_det_data = (const uint8_t *)(arguments[nargs++]);
    const uint8_t *ref_seg_data = (const uint8_t *)(arguments[nargs++]);
    int do_segment = *(int*)(arguments2[nargs++]);
    float *iou = (float*)(arguments[nargs++]);

    eval_iou(global_onnx_ctx, det_data, seg_data, ref_det_data, ref_seg_data, do_segment, iou);
}

std::string getDNNPath() {
    const char *path = getenv("POCL_DNN_DIR");

    if(NULL == path){
#ifdef __ANDROID__
        path = "/data/user/0/org.portablecl.poclaisademo/files";
#else
        path = "/tmp";
#endif
    }

    std::string ret(path);
    return ret;
}

void init_onnx(cl_program program, cl_uint device_i) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif

    if(nullptr != global_onnx_ctx) {
        POCL_MSG_PRINT_INFO("onnx init already performed once\n");
        return;
    }

    constexpr int MODEL_W = 640;
    constexpr int MODEL_H = 480;
    constexpr Task task = Task::SEGMENT;
    constexpr int MASK_W = 160;
    constexpr int MASK_H = 120;

    std::string DNNPath = getDNNPath();
    bool runOnGPU = true;

    global_onnx_ctx = new OnnxCtx(DNNPath + "/yolov8n-seg.onnx" , task,
                                  cv::Size(MODEL_W, MODEL_H),
                                  cv::Size(MASK_W, MASK_H),
                                  runOnGPU);
}

void finish_onnx(cl_device_id device, cl_program program,
                 unsigned dev_i) {
    if (global_onnx_ctx != nullptr) {
        delete global_onnx_ctx;
        global_onnx_ctx = nullptr;
    }
}

std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

cv::Mat sigmoid(const cv::Mat &input) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif
    cv::Mat output;
    cv::exp(-input, output);
    cv::divide(1, 1 + output, output);
    return output;
}

void run_onnx_inference(OnnxCtx *const onnx_ctx, const unsigned char *input, int width, int height,
                        int rotate_cw_degrees, int inp_format,
                        unsigned int *output, unsigned char *out_mask) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif

    assert(onnx_ctx);

    cv::Mat img_rgb;
    switch (inp_format) {
    case 0: {
        // Plain ol' RGB
        POCL_MSG_PRINT_INFO("DNN: already RGB, no color transform done\n");
        img_rgb = cv::Mat(height, width, CV_8UC3, (unsigned char *) input);
        break;
    }
    case 1: {
        // Android format with interleaved U/V samples
        POCL_MSG_PRINT_INFO("DNN: NV21 -> RGB transform\n");
        cv::Mat img(height + height / 2, width, CV_8UC1,
                    (unsigned char *) input);
        // For some reason, converting to BGR, not RGB, gives the correct RGB layout:
        cvtColor(img, img_rgb, cv::COLOR_YUV2BGR_NV21);
        break;
    }
    case 2: {
        // Android format with separate U/V planes
        POCL_MSG_PRINT_INFO("DNN: YV12 -> RGB transform\n");
        cv::Mat img(height + height / 2, width, CV_8UC1,
                    (unsigned char *) input);
        // For some reason, converting to BGR, not RGB, gives the correct RGB layout:
        cvtColor(img, img_rgb, cv::COLOR_YUV2BGR_YV12);
        break;
    }
    default:
        POCL_MSG_ERR(
                "DNN: Unsupported input format %d, no input transform performed.\n",
                inp_format);
    }

    switch (rotate_cw_degrees) {
        case 0:
            POCL_MSG_PRINT_INFO("DNN: No rotation\n");
        break;
        case 90:
        case -270:
            POCL_MSG_PRINT_INFO("DNN: Rotate 90 degrees clockwise\n");
        cv::rotate(img_rgb, img_rgb, cv::ROTATE_90_CLOCKWISE);
        break;
        case 180:
        case -180:
            POCL_MSG_PRINT_INFO("DNN: Rotate 180 degrees\n");
        cv::rotate(img_rgb, img_rgb, cv::ROTATE_180);
        break;
        case 270:
        case -90:
            POCL_MSG_PRINT_INFO("DNN: Rotate 90 degrees counter-clockwise\n");
        cv::rotate(img_rgb, img_rgb, cv::ROTATE_90_COUNTERCLOCKWISE);
        break;
        default:
            POCL_MSG_ERR(
                    "DNN: Unsupported rotation of %d degrees, no rotation performed.\n",
                    rotate_cw_degrees);
    }

    const int out_mask_w = img_rgb.cols / 4;  // 160 or 120
    const int out_mask_h = img_rgb.rows / 4;  // 120 or 160

    onnx_ctx->setRotationCwDegrees(rotate_cw_degrees);

    cv::Mat modelInput;
    img_rgb.convertTo(modelInput, CV_32F);

    // Letter box: Shrink image to model input size (640x480), preserving aspect ratio,
    // and fill the rest with zeros.
    float resize_scale;

    if (img_rgb.cols >= img_rgb.rows) {
        resize_scale = (float)(img_rgb.cols) / (float)(onnx_ctx->modelShape.width);
        cv::resize(modelInput, modelInput, cv::Size(onnx_ctx->modelShape.width,
          (int)(img_rgb.rows / resize_scale)));
    } else {
        resize_scale = (float)(img_rgb.rows) / (float)(onnx_ctx->modelShape.height);
        cv::resize(modelInput, modelInput,
          cv::Size((int)(img_rgb.cols / resize_scale), onnx_ctx->modelShape.height));
    }
    cv::Mat tmp_img = cv::Mat::zeros(onnx_ctx->modelShape.height, onnx_ctx->modelShape.width, CV_8UC3);
    modelInput.copyTo(tmp_img(cv::Rect(0, 0, modelInput.cols, modelInput.rows)));
    modelInput = tmp_img;

    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, onnx_ctx->modelShape,
                           cv::Scalar(), false, false);
    assert(blob.isContinuous());

    // Note: The data layout of the blob is NCHW with N=1, C=3, H=480, W=640.
    // The image channels are arranged in R,G,B order.

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> blob_size;
    for (int i = 0; i < blob.dims; ++i) {
        blob_size.push_back(blob.size[i]);
    }
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), blob_size.data(),
        blob.dims);

    // TODO: query input/output names from net
    const std::vector<const char *> input_names = {"images"};
    const std::vector<const char *> output_names = {"output0", "output1"};
    std::vector<Ort::Value> net_outputs = onnx_ctx->net->Run(
        Ort::RunOptions{}, input_names.data(), &input_tensor,
        input_names.size(), output_names.data(), output_names.size());

    // Process detection results
    auto type_info = onnx_ctx->net->GetOutputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    int dimensions = tensor_info.GetShape()[1];
    int rows = tensor_info.GetShape()[2];

    cv::Mat detection_output =
        cv::Mat(dimensions, rows, CV_32FC1,
                net_outputs[0].GetTensorMutableData<float>());
    // transpose to correct shape
    cv::transpose(detection_output, detection_output);
    float *data = reinterpret_cast<float *>(detection_output.data);

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<float *> mask_coeffs;

    for (int i = 0; i < rows; ++i) {
        float *classes_scores = data + 4;

        cv::Mat scores(1, onnx_ctx->classes.size(), CV_32FC1, classes_scores);
        cv::Point class_id;
        double maxClassScore;

        cv::minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > onnx_ctx->modelScoreThreshold) {
            confidences.push_back(maxClassScore);
            class_ids.push_back(class_id.x);

            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            unsigned int left = (unsigned int)((x - 0.5 * w) * resize_scale);
            unsigned int top = (unsigned int)((y - 0.5 * h) * resize_scale);

            unsigned int width = (unsigned int)(w * resize_scale);
            unsigned int height = (unsigned int)(h * resize_scale);

            boxes.push_back(cv::Rect(left, top, width, height));
            if (onnx_ctx->task == Task::SEGMENT) {
                mask_coeffs.push_back(data + 4 + onnx_ctx->classes.size());
            }
        }

        data += dimensions;
    }

    // Non-max suppression: Prune overlapping bounding boxes
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, onnx_ctx->modelScoreThreshold,
                      onnx_ctx->modelNMSThreshold, nms_result);

    nms_result.resize(MIN(10, nms_result.size()));
    output[0] = (unsigned int) (nms_result.size());
    POCL_MSG_PRINT_INFO("DNN: Number of detections: %ld\n", nms_result.size());

    for (unsigned int i = 0; i < nms_result.size(); ++i) {
        int idx = nms_result[i];

        // TODO: Don't write beyond MAX_DETECTIONS
        output[1 + 6 * i + 0] = (unsigned int) (class_ids[idx]);
        output[1 + 6 * i + 1] =
                *reinterpret_cast<unsigned int *>(&confidences[idx]);
        output[1 + 6 * i + 2] = (unsigned int) (boxes[idx].x);
        output[1 + 6 * i + 3] = (unsigned int) (boxes[idx].y);
        output[1 + 6 * i + 4] = (unsigned int) (boxes[idx].width);
        output[1 + 6 * i + 5] = (unsigned int) (boxes[idx].height);

        POCL_MSG_PRINT_INFO("DNN: detection %d: box %dx%d at (%d, %d), "
                            "class %d, conf %.3f\n",
                            i, boxes[idx].width, boxes[idx].height,
                            boxes[idx].x, boxes[idx].y, class_ids[idx],
                            confidences[idx]);
    }

    // Process segmentation results
    if (onnx_ctx->task == Task::SEGMENT) {
        auto type_info = onnx_ctx->net->GetOutputTypeInfo(1);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        int mask_w = tensor_info.GetShape()[3];
        int mask_h = tensor_info.GetShape()[2];

        if ((mask_w != onnx_ctx->segmentationMaskShape.width) ||
            (mask_h != onnx_ctx->segmentationMaskShape.height)) {
            POCL_MSG_ERR("DNN: Unexpected segmentation mask shape. Got %dx%d, "
                         "expected %dx%d\n",
                         mask_w, mask_h,
                         (int) (onnx_ctx->segmentationMaskShape.width),
                         (int) (onnx_ctx->segmentationMaskShape.height));

        }

        const float threshold = 0.60;

        const std::vector<int> squeezed_size{
            (int)tensor_info.GetShape()[1],
            mask_h * mask_w,
        };

        std::vector<int> intshape;
        for (int i = 0; i < tensor_info.GetShape().size(); ++i) {
            intshape.push_back((int)tensor_info.GetShape()[i]);
        }
        cv::Mat proto_squeezed = cv::Mat(
            intshape, CV_32FC1, net_outputs[1].GetTensorMutableData<float>());
        proto_squeezed = proto_squeezed.reshape(0, squeezed_size);
        int num_mask_coeffs = dimensions - onnx_ctx->classes.size() - 4;

        for (unsigned int i = 0; i < nms_result.size(); ++i) {
            int idx = nms_result[i];

            cv::Mat mc(1, num_mask_coeffs, CV_32FC1, mask_coeffs[idx]);
            cv::Mat mask = sigmoid(mc * proto_squeezed);
            cv::Mat binary_mask_1d = mask > threshold;
            cv::Mat binary_mask_2d(mask_h, mask_w, CV_8UC1, binary_mask_1d.data);

            // Undo the letter box: Crop the section occupied by the actual segmentations
            // and resize it to the output segmentation mask shape with the same aspect ratio
            // as the rotated input image.
            cv::Mat roi(binary_mask_2d, cv::Rect(0,0,
              (int)((float)(out_mask_w) / resize_scale),
              (int)((float)(out_mask_h) / resize_scale)));
            cv::Mat cropped_binary_mask;
            roi.copyTo(cropped_binary_mask);

            cv::Mat resized_binary_mask;
            cv::resize(cropped_binary_mask, resized_binary_mask,
              cv::Size(out_mask_w, out_mask_h));

            memcpy(out_mask + i * mask_w * mask_h, resized_binary_mask.data,
                   mask_w * mask_h);
        }
    }
}

void run_segmentation_postprocess(const OnnxCtx *const onnx_ctx, const unsigned int *detection_data,
                                  const unsigned char *segmentation_data,
                                  unsigned char *output) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif
    assert(onnx_ctx);

    unsigned int num_detections = detection_data[0];
    // TODO: The mask shapes should be set via the out_mask_w/h defined in run_onnx_inference()
    int mask_w = onnx_ctx->segmentationMaskShape.width;
    int mask_h = onnx_ctx->segmentationMaskShape.height;
    int img_w = onnx_ctx->modelShape.width;
    int img_h = onnx_ctx->modelShape.height;

    if (onnx_ctx->rotationCwDegrees % 180 != 0) {
      mask_h = onnx_ctx->segmentationMaskShape.width;
      mask_w = onnx_ctx->segmentationMaskShape.height;
      img_h = onnx_ctx->modelShape.width;
      img_w = onnx_ctx->modelShape.height;
    }

    cv::Mat color_mask(mask_h, mask_w, CV_8UC1, cv::Scalar(NO_CLASS_ID));

    for (int i = 0; i < num_detections; ++i) {
        unsigned char class_id = (unsigned char)(detection_data[1 + 6 * i]);

        int box_x =
            (int)((float)(detection_data[1 + 6 * i + 2]) / (float)(img_w) * (float)(mask_w));
        int box_y =
            (int)((float)(detection_data[1 + 6 * i + 3]) / (float)(img_h) * (float)(mask_h));
        int box_w =
            (int)((float)(detection_data[1 + 6 * i + 4]) / (float)(img_w) * (float)(mask_w));
        int box_h =
            (int)((float)(detection_data[1 + 6 * i + 5]) / (float)(img_h) * (float)(mask_h));

        box_x = std::min(std::max(box_x, 0), mask_w);
        box_y = std::min(std::max(box_y, 0), mask_h);
        box_w = std::min(box_w, mask_w - box_x);
        box_h = std::min(box_h, mask_h - box_y);

        if (box_w > 0 && box_h > 0) {
            cv::Mat raw_mask(mask_h, mask_w, CV_8UC1,
                             (void *)(segmentation_data + i * mask_w * mask_h));
            cv::Rect roi(box_x, box_y, box_w, box_h);
            cv::Mat raw_mask_roi = cv::Mat::zeros(mask_h, mask_w, CV_8UC1);
            raw_mask(roi).copyTo(raw_mask_roi(roi));
            color_mask.setTo(cv::Scalar(class_id), raw_mask_roi);
        }
    }

    memcpy(output, color_mask.data, mask_w * mask_h);
}

void run_segmentation_reconstruct(const OnnxCtx *const onnx_ctx,
                                  const unsigned char *postprocess_data,
                                  unsigned char *output) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif
    assert(onnx_ctx);

    const int nsamples = onnx_ctx->segmentationMaskShape.width * onnx_ctx->segmentationMaskShape.height;

    // map class indices to colors
    // TODO: This can be done with cv::applyColorMap but I didn't get it to work.
    for (int i = 0; i < nsamples; ++i) {
        const uint8_t class_id = postprocess_data[i];

        const int color_int = SEGMENTATION_COLORS[class_id];
        const unsigned char *channels =
            reinterpret_cast<const unsigned char *>(&color_int);

        // RGBA image
        output[4 * i] = channels[0] / 2;
        output[4 * i + 1] = channels[1] / 2;
        output[4 * i + 2] = channels[2] / 2;
        output[4 * i + 3] = channels[3] / 2;
    }
}

// TODO: remove this
void run_decompress_from_jpeg_rgb888(const uint8_t *input,
                                     const uint64_t *input_size, int32_t width,
                                     int32_t height, uint8_t *output) {
  assert(*input_size <= INT_MAX);
#ifdef TRACY_ENABLE
  ZoneScoped;
#endif
  cv::Mat data_mat(1, *input_size, CV_8UC1, (void *) input);

    cv::Mat decoded_data = imdecode(data_mat, cv::IMREAD_COLOR);

  if (NULL != decoded_data.data) {
    // paranoid check to make sure that the data is actually the size
    // we expect it to be.
    size_t data_size = decoded_data.dataend - decoded_data.datastart;
    data_size = data_size < 0 ? -data_size : data_size;
    assert( data_size == width * height * 3);

    memcpy (output, decoded_data.data, width * height * 3);
  } else {
      POCL_MSG_ERR("JPEG DECOMPRESS: Decoded data is NULL!\n");
  }
}

void eval_iou(const OnnxCtx *const onnx_ctx,
              const uint8_t *det_data, const uint8_t *seg_data,
              const uint8_t *ref_det_data, const uint8_t *ref_seg_data,
              int do_segment, float *iou) {
#ifdef TRACY_ENABLE
    ZoneScoped;
#endif

    assert(onnx_ctx);

    if (do_segment) {
        uint16_t correct[NUM_CLASSES] = {0};
        uint16_t wrong[NUM_CLASSES] = {0};

        const int npx = onnx_ctx->segmentationMaskShape.width
                        * onnx_ctx->segmentationMaskShape.height;

        POCL_MSG_PRINT_INFO("EVAL IOU: num_pixels: %d, num_classes: %d\n", npx,
                            NUM_CLASSES);

        for (int i = 0; i < npx; ++i) {
            int ground_truth_class = ref_seg_data[i];
            int predicted_class = seg_data[i];

            if (predicted_class >= NUM_CLASSES || ground_truth_class >= NUM_CLASSES) {
                POCL_MSG_ERR(
                    "EVAL IOU: Pixel %d, invalid class predicted: %d, ground truth: %d. Must be < %d. Setting IoU to -6.0 and skipping.\n",
                    i, predicted_class, ground_truth_class, NUM_CLASSES
                );

                *iou = -6.0f;
                return;
            }

            if (ground_truth_class == NO_CLASS_ID) {
                if (predicted_class == NO_CLASS_ID) {
                    // correct no prediction, true negative, not interested
                } else {
                    // false positive
                    wrong[predicted_class] += 1;
                }
            } else {
                if (ground_truth_class == predicted_class) {
                    // true positive
                    correct[predicted_class] += 1;
                } else if (predicted_class == NO_CLASS_ID) {
                    // false negative
                    wrong[ground_truth_class] += 1;
                } else {
                    // false positive
                    wrong[predicted_class] += 1;
                }
            }
        }

        float tmp_iou = 0.0f;
        int num_detections = 0;

        for (int cls = 0; cls < NUM_CLASSES; ++cls) {
            if ((correct[cls] + wrong[cls]) != 0) {
                tmp_iou += (float) (correct[cls])
                           / (float) (correct[cls] + wrong[cls]);
                num_detections += 1;

                POCL_MSG_PRINT_INFO(
                        "EVAL IOU: class %3d (%15s), correct: %5d, wrong: %5d, iou: %5.3f\n",
                        cls, onnx_ctx->classes[cls].c_str(), correct[cls],
                        wrong[cls], tmp_iou
                );
            }
        }

        if (num_detections == 0) {
            *iou = -2.0f;
        } else {
            *iou = tmp_iou / (float) (num_detections);
            POCL_MSG_PRINT_INFO("EVAL IOU: %5.3f\n", *iou);
        }
    } else {
        *iou = -3.0f;
    }
}

void _pocl_kernel_pocl_dnn_ctx_init_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

#ifdef TRACY_ENABLE
    ZoneScoped;
#endif

    if( nullptr != *((pocl_context *)context)->data) {
        POCL_MSG_PRINT_INFO("onnx ctx init already performed once\n");
        return;
    }

    void **arguments = *(void ***) (args);

    // TODO: remove this
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[0]);



    constexpr int MODEL_W = 640;
    constexpr int MODEL_H = 480;
    constexpr int IMG_W = 640;
    constexpr int IMG_H = 480;
    constexpr Task task = Task::SEGMENT;
    constexpr int MASK_W = 160;
    constexpr int MASK_H = 120;

// TODO: get the path from an env variable instead of hardcoding this
#ifdef __ANDROID__
    std::string projectBasePath = "/data/user/0/org.portablecl.poclaisademo/files";
#else
    std::string projectBasePath = "/tmp";
#endif

    bool runOnGPU = true;

    *((pocl_context *)context)->data = new OnnxCtx(projectBasePath + "/yolov8n-seg.onnx", task,
                       cv::Size(MODEL_W, MODEL_H), cv::Size(MASK_W, MASK_H),
                       runOnGPU);

}

void _pocl_kernel_pocl_dnn_ctx_destroy_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {
// TODO: remove this function


//    void **arguments = *(void ***) (args);
//
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[0]);
//
//    if (NULL != *ctx) {
//        delete *ctx;
//        *ctx = NULL;
//    }
}

void _pocl_kernel_pocl_dnn_ctx_detection_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

    // TODO: get program data from pocl_context
    void **arguments = *(void ***) (args);
    void **arguments2 = (void **) (args);

    int nargs = 0;
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[nargs++]);
    nargs++;
    OnnxCtx *ctx = (OnnxCtx *) *(((pocl_context *)context)->data);
    const unsigned char *data = (const unsigned char *) (arguments[nargs++]);
    int width = *(int *) (arguments2[nargs++]);
    int height = *(int *) (arguments2[nargs++]);
    int rotate_cw_degrees = *(int *) (arguments2[nargs++]);
    int inp_format = *(int *) (arguments2[nargs++]);
    unsigned int *output = (unsigned int *) (arguments[nargs++]);
    unsigned char *out_mask = (unsigned char *) (arguments[nargs++]);

    run_onnx_inference(ctx, data, width, height, rotate_cw_degrees, inp_format,
                       output, out_mask);
}

void _pocl_kernel_pocl_dnn_ctx_segmentation_postprocess_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {
    void **arguments = *(void ***) (args);

    int nargs = 0;
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[nargs++]);
    nargs++;
    OnnxCtx *ctx = (OnnxCtx *) *(((pocl_context *)context)->data);
    const unsigned int *detection_data = (const unsigned int *) (arguments[nargs++]);
    const unsigned char *segmentation_data = (const unsigned char *) (arguments[nargs++]);
    unsigned char *output = (unsigned char *) (arguments[nargs++]);

    run_segmentation_postprocess(ctx, detection_data, segmentation_data, output);
}

void _pocl_kernel_pocl_dnn_ctx_segmentation_reconstruct_u8_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

    void **arguments = *(void ***) (args);

    int nargs = 0;
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[nargs++]);
    nargs++;
    OnnxCtx *ctx = (OnnxCtx *) *(((pocl_context *)context)->data);
    const unsigned char *postprocess_data = (const unsigned char *) (arguments[nargs++]);
    unsigned char *output = (unsigned char *) (arguments[nargs++]);

    run_segmentation_reconstruct(ctx, postprocess_data, output);
}

void _pocl_kernel_pocl_dnn_ctx_eval_iou_f32_workgroup(
        cl_uchar *args, cl_uchar *context,
        ulong group_x, ulong group_y,
        ulong group_z) {

    void **arguments = *(void ***) (args);
    void **arguments2 = (void **) (args);

    int nargs = 0;
//    OnnxCtx **ctx = (OnnxCtx **) (arguments[nargs++]);
    nargs++;
    OnnxCtx *ctx = (OnnxCtx *) *(((pocl_context *)context)->data);
    const uint8_t *det_data = (const uint8_t *) (arguments[nargs++]);
    const uint8_t *seg_data = (const uint8_t *) (arguments[nargs++]);
    const uint8_t *ref_det_data = (const uint8_t *) (arguments[nargs++]);
    const uint8_t *ref_seg_data = (const uint8_t *) (arguments[nargs++]);
    int do_segment = *(int *) (arguments2[nargs++]);
    float *iou = (float *) (arguments[nargs++]);

    eval_iou(ctx, det_data, seg_data, ref_det_data, ref_seg_data, do_segment, iou);
}

void init_onnx_ctx(cl_program program, cl_uint device_i) {

    // TODO: see if this is needed

//#ifdef TRACY_ENABLE
//    ZoneScoped;
//#endif
//    // if the onnx context has already been initialized by another kernel,
//    // don't do anything.
//    if(nullptr != program->data[device_i]){
//        return;
//    }
//
//    constexpr int MODEL_W = 640;
//    constexpr int MODEL_H = 480;
//    constexpr int IMG_W = 640;
//    constexpr int IMG_H = 480;
//    constexpr Task task = Task::SEGMENT;
//    constexpr int MASK_W = 160;
//    constexpr int MASK_H = 120;
//
//// TODO: get the path from an env variable instead of hardcoding this
//#ifdef __ANDROID__
//    std::string projectBasePath = "/data/user/0/org.portablecl.poclaisademo/files";
//#else
//    std::string projectBasePath = "/tmp";
//#endif
//
//    bool runOnGPU = true;
//
//    program->data[device_i] = new OnnxCtx(projectBasePath + "/yolov8n-seg.onnx", task,
//                                  cv::Size(MODEL_W, MODEL_H), cv::Size(MASK_W, MASK_H),
//                                  runOnGPU);
}

void finish_onnx_ctx(cl_device_id device, cl_program program, unsigned dev_i) {

    if (program->data[dev_i] != nullptr) {
        delete program->data[dev_i];
        program->data[dev_i] = nullptr;
    }
}

POCL_EXPORT
void kick_onnx_awake() {
    POCL_MSG_PRINT_INFO("kicking onnx awake\n");
    init_onnx(NULL, NULL);
}
