#ifndef RecoTracker_LSTCore_src_alpaka_T5NeuralNetworkWeights_h
#define RecoTracker_LSTCore_src_alpaka_T5NeuralNetworkWeights_h

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/HostDeviceConstant.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::dnn::t5dnn {
#ifdef LST_T5DNN_RETRAINED
#include "T5NeuralNetworkWeights_retrained.h"
#else
#include "T5NeuralNetworkWeights_original.h"
#endif
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst::dnn::t5dnn

#endif
