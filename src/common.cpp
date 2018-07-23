#ifndef _COMMON_CPP
#define _COMMON_CPP

#include "gradient.cpp"

namespace NeuralNetwork {

#define LEARNING_RATE 0.02
#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.001

struct point_tensor
{
        int x;
        int y;
        int z;
};

struct size_tensor
{
        int width;
        int height;
        int depth;
};

struct range_tensor
{
        int min_x, min_y, min_z;
        int max_x, max_y, max_z;
};

static float update_weight(float w, Gradient* grad, float multp = 1)
{
        float m = (grad->grad + grad->oldgrad * MOMENTUM);
        w -= LEARNING_RATE * m * multp +
             LEARNING_RATE * WEIGHT_DECAY * w;
        return w;
}

static void update_gradient(Gradient* grad)
{
        grad->oldgrad = (grad->grad + grad->oldgrad * MOMENTUM);
}

}

#endif
