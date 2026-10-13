// nvinfer custom bbox parser for the NMS-free YOLO26 ONNX export: output [N, 300, 6] = x1 y1 x2 y2 conf class
// (coordinates in network-input pixels; nvinfer rescales them to the frame). All classes of the model are kept
// (fine-tuned model: 0 human, 1 bike, 2 vehicle); num-detected-classes in the pgie config must equal the class count.
#include <algorithm>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool NvDsInferParseYolo26(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                     NvDsInferNetworkInfo const &networkInfo,
                                     NvDsInferParseDetectionParams const &detectionParams,
                                     std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo &layer = outputLayersInfo[0];
    if (layer.inferDims.numDims < 2 || layer.inferDims.d[1] != 6) {
        std::cerr << "NvDsInferParseYolo26: expected output [300,6] (NMS-free export), got a different shape\n";
        return false;
    }
    const float *p = static_cast<const float *>(layer.buffer);
    const unsigned n = layer.inferDims.d[0];
    const float W = networkInfo.width, H = networkInfo.height;

    for (unsigned i = 0; i < n; ++i, p += 6) {
        const int cls = static_cast<int>(p[5]);
        const float conf = p[4];
        if (cls < 0 || cls >= static_cast<int>(detectionParams.numClassesConfigured)) continue;
        if (conf < detectionParams.perClassPreclusterThreshold[cls]) continue;
        const float x1 = std::min(std::max(p[0], 0.f), W), y1 = std::min(std::max(p[1], 0.f), H);
        const float x2 = std::min(std::max(p[2], 0.f), W), y2 = std::min(std::max(p[3], 0.f), H);
        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;
        NvDsInferObjectDetectionInfo o{};  // zero-init: newer headers may add fields (e.g. an angle) we don't set
        o.classId = cls;
        o.detectionConfidence = conf;
        o.left = x1;
        o.top = y1;
        o.width = x2 - x1;
        o.height = y2 - y1;
        objectList.push_back(o);
    }
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);