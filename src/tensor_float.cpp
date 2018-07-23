#ifndef _TENSOR_FLOAT_CPP
#define _TENSOR_FLOAT_CPP

#include <cassert>
#include <iostream>
#include "tensor.cpp"

namespace NeuralNetwork {

class TensorFloat : public Tensor {

public:

float *values = NULL;

TensorFloat() {

}

TensorFloat(int width, int height, int depth) {
        values = new float[width * height * depth];
        size.width = width;
        size.height = height;
        size.depth = depth;
}

TensorFloat(const TensorFloat& t) {
        values = new float[t.size.width * t.size.height * t.size.depth];
        memcpy(this->values, t.values, t.size.width * t.size.height * t.size.depth * sizeof(float));
        this->size = t.size;
}

static TensorFloat* diff(TensorFloat *tensor_a, TensorFloat *tensor_b) {

        TensorFloat* clone = new TensorFloat(*tensor_a);
        for(int i = 0; i < tensor_b->size.width * tensor_b->size.height * tensor_b->size.depth; i++) {
                clone->values[i] -= tensor_b->values[i];
        }
        return clone;

}

float& operator()(int x, int y, int z) const
{
        return this->get( x, y, z );
}

float& get(int x, int y, int z) const
{
        assert(x >= 0 && y >= 0 && z >= 0);
        assert(x < size.width && y < size.height && z < size.depth);
        return values[z * (size.width * size.height) + y * size.width + x];
}

~TensorFloat() {
        if(values != NULL) {
                delete[] values;
        }
}

};

}

#endif
