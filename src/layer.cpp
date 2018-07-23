#ifndef _LAYER_CPP
#define _LAYER_CPP

#include "common.cpp"
#include "tensor_float.cpp"
#include "layer_grid_frame_buffer.cpp"

namespace NeuralNetwork {

enum LayerType { convolutional, fc, relu, pool, dropout_layer };

// Layer abstract class
class Layer {

public:

LayerType type;
TensorFloat *input_gradients;
TensorFloat *input;
TensorFloat *output;
size_tensor input_size;
size_tensor output_size;
LayerGridFrameBuffer *gridRenderFrameBuffer;

virtual void activate(TensorFloat*)=0;
virtual void activate()=0;
virtual void calc_grads(TensorFloat*)=0;
virtual void fix_weights()=0;

};

}

#endif
