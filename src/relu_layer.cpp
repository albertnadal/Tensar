#ifndef _RELU_LAYER_CPP
#define _RELU_LAYER_CPP

#include "layer.cpp"
#include "tensor_float.cpp"
#include "layer_grid_frame_buffer.cpp"
#include "tensor_render_frame_buffer.cpp"

namespace NeuralNetwork {

class ReLuLayer : public Layer {

public:

ReLuLayer(size_tensor in_size) {
        type = LayerType::relu;
        input_size = in_size;

        //Define and alloc a grid layout for render buffers

        // {input, output}
        // {cellObj, cellObj}

        gridRenderFrameBuffer = new LayerGridFrameBuffer(2, in_size.depth, "ReLu");   // 2 = {input, output}

        // Initialize first column of the grid with Inputs buffers
        gridRenderFrameBuffer->column_titles.push_back("in");
        char* subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height, in_size.depth);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<in_size.depth; i++) {
                gridRenderFrameBuffer->set(0, i, new TensorRenderFrameBuffer(in_size.width, in_size.height));
        }

        // Initialize forth column of the grid with Output buffers
        gridRenderFrameBuffer->column_titles.push_back("out");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<in_size.depth; i++) {
                gridRenderFrameBuffer->set(1, i, new TensorRenderFrameBuffer(in_size.width, in_size.height));
        }

        input_gradients = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        input = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        output = new TensorFloat(in_size.width, in_size.height, in_size.depth);
}

void activate(TensorFloat *in) {
        this->input = in;

        for(int z = 0; z < in->size.depth; z++)
        {
                // Update render frame input buffer values
                TensorRenderFrameBuffer* inputFrameBuffer = gridRenderFrameBuffer->get(0, z);
                for(int x = 0; x < in->size.width; x++)
                {
                        for(int y = 0; y < in->size.height; y++)
                        {
                                float value = in->get(x, y, z);
                                inputFrameBuffer->set(x, y, (int)(value * 255));
                        }
                }
                inputFrameBuffer->swapBuffers();
        }

        // Activate
        activate();
}

void activate() {

        for(int z = 0; z < input->size.depth; z++)
        {
                TensorRenderFrameBuffer* outputFrameBuffer = gridRenderFrameBuffer->get(1, z);
                for(int x = 0; x < input->size.width; x++)
                {
                        for(int y = 0; y < input->size.height; y++)
                        {

                                float value = (*input)(x, y, z);   //in(x, y, z);
                                if(value < 0)
                                        value = 0;
                                (*output)(x, y, z) = value;
                                outputFrameBuffer->set(x, y, (int)(value * 255));
                        }
                }
                outputFrameBuffer->swapBuffers();
        }
}

void fix_weights() {

}

void calc_grads(TensorFloat* grad_next_layer) {

        for(int x = 0; x < input->size.width; x++)
        {
                for(int y = 0; y < input->size.height; y++)
                {
                        for(int z = 0; z < input->size.depth; z++)
                        {
                                (*input_gradients)(x, y, z) = ((*input)(x, y, z) < 0) ? 0 : (1 * (*grad_next_layer)(x, y, z));
                        }
                }
        }

}

~ReLuLayer() {
        delete gridRenderFrameBuffer;
        delete input_gradients;
        delete input;
        delete output;
}

};

}

#endif
