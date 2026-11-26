// nvinfer custom bbox parser for the NMS-free YOLO26 ONNX export: output [N, 300, 6] = x1 y1 x2 y2 conf class
// (coordinates in network-input pixels; nvinfer rescales them to the frame). All classes of the model are kept
// (fine-tuned model: 0 human, 1 bike, 2 vehicle); num-detected-classes in the pgie config must equal the class count.
//
// Robust to both ways nvinfer may report the layer dims:  [300,6]  or  [B,300,6] / [-1,300,6].
// nvinfer calls this function once per frame, with layer.buffer pointing at that frame's [300,6] block.
#include <algorithm>
#include <atomic>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

static void printLayers(std::vector<NvDsInferLayerInfo> const &layers)
{
    std::cerr << "NvDsInferParseYolo26: " << layers.size() << " output layer(s)\n";
    for (size_t k = 0; k < layers.size(); ++k) {
        const NvDsInferDims &d = layers[k].inferDims;
        std::cerr << "  [" << k << "] '" << (layers[k].layerName ? layers[k].layerName : "?") << "' dims=[";
        for (unsigned i = 0; i < d.numDims; ++i) std::cerr << d.d[i] << (i + 1 < d.numDims ? "," : "");
        std::cerr << "]\n";
    }
}

extern "C" bool NvDsInferParseYolo26(std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
                                     NvDsInferNetworkInfo const &networkInfo,
                                     NvDsInferParseDetectionParams const &detectionParams,
                                     std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    static std::atomic<bool> reported{false};

    if (outputLayersInfo.empty()) {
        std::cerr << "NvDsInferParseYolo26: no output layers\n";
        return false;
    }

    // Pick the output whose last dimension is 6 (x1 y1 x2 y2 conf class); normally it is layer 0.
    const NvDsInferLayerInfo *layer = nullptr;
    for (const auto &l : outputLayersInfo) {
        const unsigned nd = l.inferDims.numDims;
        if (nd >= 2 && l.inferDims.d[nd - 1] == 6) {
            layer = &l;
            break;
        }
    }
    if (!layer) {
        if (!reported.exchange(true)) {
            std::cerr << "NvDsInferParseYolo26: no output layer ending in 6 (expected NMS-free [300,6])\n";
            printLayers(outputLayersInfo);
        }
        return false;
    }
    if (layer->buffer == nullptr) {
        std::cerr << "NvDsInferParseYolo26: output buffer is null\n";
        return false;
    }

    const unsigned nd = layer->inferDims.numDims;
    const int dn = static_cast<int>(layer->inferDims.d[nd - 2]);  // number of detections per frame (300)
    if (dn <= 0) {
        std::cerr << "NvDsInferParseYolo26: invalid detection count " << dn << "\n";
        return false;
    }
    const unsigned n = static_cast<unsigned>(dn);

    if (!reported.exchange(true)) {  // one-time info line, helps to verify the layout
        std::cerr << "NvDsInferParseYolo26: using layer '" << (layer->layerName ? layer->layerName : "?")
                  << "' dims=[";
        for (unsigned i = 0; i < nd; ++i) std::cerr << layer->inferDims.d[i] << (i + 1 < nd ? "," : "");
        std::cerr << "], " << n << " candidates per frame, classes configured="
                  << detectionParams.numClassesConfigured << "\n";
    }

    const float *p = static_cast<const float *>(layer->buffer);
    const float W = static_cast<float>(networkInfo.width), H = static_cast<float>(networkInfo.height);
    const int ncls = static_cast<int>(
        std::min<size_t>(detectionParams.numClassesConfigured, detectionParams.perClassPreclusterThreshold.size()));

    for (unsigned i = 0; i < n; ++i, p += 6) {
        const int cls = static_cast<int>(p[5]);
        const float conf = p[4];
        if (cls < 0 || cls >= ncls) continue;
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