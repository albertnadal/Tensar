#ifndef _INPUT_CASE_CPP
#define _INPUT_CASE_CPP

#include "tensor_float.cpp"

namespace NeuralNetwork {

class InputCase {

public:

TensorFloat *data;
TensorFloat *output;

InputCase(size_tensor data_size, size_tensor out_size) {
        data = new TensorFloat(data_size.width, data_size.height, data_size.depth);
        output = new TensorFloat(out_size.width, out_size.height, out_size.depth);
}

~InputCase() {
        delete data;
        delete output;
}

};

}

#endif
