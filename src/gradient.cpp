#ifndef _GRADIENT_CPP
#define _GRADIENT_CPP

namespace NeuralNetwork {

class Gradient {

public:

float grad;
float oldgrad;
Gradient()
{
        grad = 0;
        oldgrad = 0;
}
};

}

#endif
