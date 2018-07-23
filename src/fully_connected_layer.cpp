#ifndef _FULLY_CONNECTED_LAYER_CPP
#define _FULLY_CONNECTED_LAYER_CPP

#include <vector>
#include <cmath>
#include "layer.cpp"
#include "tensor_float.cpp"
#include "layer_grid_frame_buffer.cpp"
#include "tensor_render_frame_buffer.cpp"

namespace NeuralNetwork {

class FullyConnectedLayer : public Layer {

public:

TensorFloat *weights;
vector<float> input_vector;
vector<Gradient> gradients;

FullyConnectedLayer(size_tensor in_size, size_tensor out_size) {
        type = LayerType::fc;
        input_size = in_size;
        output_size = out_size;

        //Define and alloc a grid layout for render buffers

        // {input, gradient, output}
        // {cellObj, cellObj, cellObj}

        gridRenderFrameBuffer = new LayerGridFrameBuffer(3, 1, "FullyConnected");   // 3 = {input, gradient, output}

        // Initialize first column of the grid with Inputs buffers
        gridRenderFrameBuffer->column_titles.push_back("in");
        char *subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);
        gridRenderFrameBuffer->set(0, 0, new TensorRenderFrameBuffer(in_size.width, in_size.height));


        // Initialize third column of the grid with Gradients buffers
        gridRenderFrameBuffer->column_titles.push_back("grad");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);
        gridRenderFrameBuffer->set(1, 0, new TensorRenderFrameBuffer(in_size.width, in_size.height));


        // Initialize forth column of the grid with Output buffers
        gridRenderFrameBuffer->column_titles.push_back("out");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", out_size.width, out_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);
        gridRenderFrameBuffer->set(2, 0, new TensorRenderFrameBuffer(out_size.width, out_size.height));


        input_gradients = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        gradients = vector<Gradient>(output_size.width);
        input_vector = vector<float>(output_size.width);
        input = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        output = new TensorFloat(out_size.width, out_size.height, out_size.depth);
        weights = new TensorFloat(in_size.width * in_size.height * in_size.depth, out_size.width, out_size.height);

        int maxval = in_size.width * in_size.height * in_size.depth;

        for(int i = 0; i < out_size.width; i++) {
                for(int h = 0; h < in_size.width * in_size.height * in_size.depth; h++) {
                        (*weights)(h, i, 0) = 2.19722f / maxval * rand() / float( RAND_MAX );
                }
        }
        // 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]

}

float activator_function(float x)
{
        //return tanhf( x );
        float sig = 1.0f / (1.0f + exp( -x ));
        return sig;
}

float activator_derivative(float x)
{
        //float t = tanhf( x );
        //return 1 - t * t;
        float sig = 1.0f / (1.0f + exp( -x ));
        return sig * (1 - sig);
}

int map(point_tensor d)
{
        return d.z * (input->size.width * input->size.height) + d.y * (input->size.width) + d.x;
}

void activate(TensorFloat *in) {

        this->input = in;

        // Update render frame inputs buffer values
        TensorRenderFrameBuffer* inputFrameBuffer = gridRenderFrameBuffer->get(0, 0);
        for(int x = 0; x < in->size.width; x++)
        {
                for(int y = 0; y < in->size.height; y++)
                {
                        for(int z = 0; z < in->size.depth; z++)
                        {
                                float value = in->get(x, y, z);
                                inputFrameBuffer->set(x, y, (int)(value * 255));
                        }
                }
        }

        inputFrameBuffer->swapBuffers();

        // Activate
        activate();
}

void activate() {

        TensorRenderFrameBuffer* outputFrameBuffer = gridRenderFrameBuffer->get(2, 0);
        for(int n = 0; n < output->size.width; n++)
        {
                float inputv = 0;
                for(int i = 0; i < input->size.width; i++)
                {
                        for(int j = 0; j < input->size.height; j++)
                        {
                                for(int z = 0; z < input->size.depth; z++)
                                {
                                        int m = map( { i, j, z } );
                                        inputv += (*input)(i, j, z) * (*weights)(m, n, 0);
                                }
                        }
                }

                input_vector[n] = inputv;
                float value = activator_function(inputv);
                (*output)(n, 0, 0) = value;
                outputFrameBuffer->set(n, 0, (int)(value * 255));
        }
        outputFrameBuffer->swapBuffers();
}

void fix_weights() {

        for(int n = 0; n < output->size.width; n++) {

                Gradient &grad = gradients[n];

                for(int i = 0; i < input->size.width; i++) {
                        for(int j = 0; j < input->size.height; j++) {
                                for(int z = 0; z < input->size.depth; z++) {
                                        int m = map( { i, j, z } );
                                        float &w = (*weights)(m, n, 0);
                                        w = update_weight(w, &grad, (*input)(i, j, z));
                                }
                        }
                }

                update_gradient(&grad);
        }

}

void calc_grads(TensorFloat* grad_next_layer) {

        memset(input_gradients->values, 0, input_gradients->size.width * input_gradients->size.height * input_gradients->size.depth * sizeof(float));
        for(int n = 0; n < output->size.width; n++)
        {
                Gradient& grad = gradients[n];
                grad.grad = (*grad_next_layer)(n, 0, 0) * activator_derivative(input_vector[n]);

                for(int i = 0; i < input->size.width; i++) {
                        for(int j = 0; j < input->size.height; j++) {
                                for(int z = 0; z < input->size.depth; z++) {
                                        int m = map( { i, j, z } );
                                        (*input_gradients)(i, j, z) += grad.grad * (*weights)(m, n, 0);
                                }
                        }
                }

        }

}

~FullyConnectedLayer() {

        delete gridRenderFrameBuffer;
        delete input_gradients;
        delete input;
        delete output;
}

};

}

#endif
