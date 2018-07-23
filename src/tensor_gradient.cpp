#ifndef _TENSOR_GRADIENT_CPP
#define _TENSOR_GRADIENT_CPP

#include <cassert>
#include "tensor.cpp"
#include "gradient.cpp"

namespace NeuralNetwork {

class TensorGradient : public Tensor {

public:

Gradient **values;

TensorGradient(int width, int height, int depth) {
        values = new Gradient*[width * height * depth];
        for(int i=0; i < width * height * depth; i++) {
                values[i] = new Gradient();
                values[i]->grad = 0;
                values[i]->oldgrad = 0;
        }
        size.width = width;
        size.height = height;
        size.depth = depth;
}

TensorGradient(const TensorGradient* t) {
        values = new Gradient*[t->size.width * t->size.height * t->size.depth];
        for(int i=0; i < t->size.width * t->size.height * t->size.depth; i++) {
                values[i] = new Gradient();
                values[i]->grad = t->values[i]->grad;
                values[i]->oldgrad = t->values[i]->oldgrad;
        }
        this->size = t->size;
}

Gradient* operator()(int x, int y, int z)
{
        return this->get(x, y, z);
}

Gradient* get(int x, int y, int z)
{
        assert(x >= 0 && y >= 0 && z >= 0);
        assert(x < size.width && y < size.height && z < size.depth);
        return values[z * (size.width * size.height) + y * size.width + x];
}

~TensorGradient() {

        for(int i=0; i < this->size.width * this->size.height * this->size.depth; i++) {
                delete values[i];
        }
        delete[] values;
}
};

}

#endif
