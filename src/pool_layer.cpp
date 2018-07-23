#ifndef _POOL_LAYER_CPP
#define _POOL_LAYER_CPP

#include <cmath>
#include <float.h>
#include <cassert>
#include <vector>
#include "layer.cpp"
#include "tensor_float.cpp"
#include "tensor_gradient.cpp"
#include "layer_grid_frame_buffer.cpp"
#include "tensor_render_frame_buffer.cpp"

namespace NeuralNetwork {

class PoolLayer : public Layer {

public:

vector<TensorGradient*> filter_gradients;
int stride, extend_filter;

PoolLayer(int stride, int extend_filter, size_tensor in_size) {
        type = LayerType::pool;
        input_size = in_size;

        //Define and alloc a grid layout for render buffers

        // {input, gradient, output}
        // {cellObj, cellObj, cellObj}

        gridRenderFrameBuffer = new LayerGridFrameBuffer(3, in_size.depth, "Pool");   // 3 = {input, gradient, output}

        // Initialize first column of the grid with Inputs buffers
        gridRenderFrameBuffer->column_titles.push_back("in");
        char* subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<in_size.depth; i++) {
                gridRenderFrameBuffer->set(0, i, new TensorRenderFrameBuffer(in_size.width, in_size.height));
        }

        // Initialize third column of the grid with Gradients buffers
        gridRenderFrameBuffer->column_titles.push_back("grad");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<in_size.depth; i++) {
                gridRenderFrameBuffer->set(1, i, new TensorRenderFrameBuffer(in_size.width, in_size.height));
        }

        // Initialize forth column of the grid with Output buffers
        gridRenderFrameBuffer->column_titles.push_back("out");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", (in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<in_size.depth; i++) {
                gridRenderFrameBuffer->set(2, i, new TensorRenderFrameBuffer((in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1));
        }

        input_gradients = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        input = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        output = new TensorFloat((in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1, in_size.depth);
        this->stride = stride;
        this->extend_filter = extend_filter;
        assert( (float( in_size.width - extend_filter ) / stride + 1) == ((in_size.width - extend_filter) / stride + 1) );
        assert( (float( in_size.height - extend_filter ) / stride + 1) == ((in_size.height - extend_filter) / stride + 1) );
}

point_tensor map_to_input(point_tensor out, int z) {
        out.x *= stride;
        out.y *= stride;
        out.z = z;
        return out;
}

int normalize_range(float f, int max, bool lim_min) {
        if(f <= 0) { return 0; }

        max -= 1;

        if(f >= max) { return max; }

        if(lim_min) {
                // left side of inequality
                return ceil( f );
        } else {
                return floor( f );
        }
}

range_tensor map_to_output(int x, int y) {
        float a = x;
        float b = y;
        return {
                       normalize_range( (a - extend_filter + 1) / stride, output->size.width, true ),
                       normalize_range( (b - extend_filter + 1) / stride, output->size.height, true ),
                       0,
                       normalize_range( a / stride, output->size.width, false ),
                       normalize_range( b / stride, output->size.height, false ),
                       (int)output->size.depth - 1,
        };
}

void activate(TensorFloat *in) {
        this->input = in;

        for(int z = 0; z < in->size.depth; z++)
        {
                // Update render frame inputs buffer values
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

        for(int z = 0; z < output->size.depth; z++)
        {
                TensorRenderFrameBuffer* outputFrameBuffer = gridRenderFrameBuffer->get(2, z);
                for(int x = 0; x < output->size.width; x++)
                {
                        for(int y = 0; y < output->size.height; y++)
                        {
                                point_tensor mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
                                float mval = -FLT_MAX;
                                for(int i = 0; i < extend_filter; i++)
                                        for(int j = 0; j < extend_filter; j++)
                                        {
                                                float v = (*input)(mapped.x + i, mapped.y + j, z);
                                                if(v > mval)
                                                        mval = v;
                                        }
                                (*output)(x, y, z) = mval;
                                outputFrameBuffer->set(x, y, (int)(mval * 255));
                        }
                }
                outputFrameBuffer->swapBuffers();
        }

}

void fix_weights() {

}

void calc_grads(TensorFloat* grad_next_layer) {

        for(int y = 0; y < input_size.height; y++)
        {
                for(int x = 0; x < input_size.width; x++)
                {
                        range_tensor rn = map_to_output(x, y);
                        for(int z = 0; z < input_size.depth; z++)
                        {
                                TensorRenderFrameBuffer* gradientFrameBuffer = gridRenderFrameBuffer->get(1, z);
                                float sum_error = 0;
                                for(int i = rn.min_x; i <= rn.max_x; i++)
                                {
                                        int minx = i * stride;
                                        for(int j = rn.min_y; j <= rn.max_y; j++)
                                        {
                                                int miny = j * stride;
                                                int is_max = (*input)(x, y, z) == (*output)(i, j, z) ? 1 : 0;
                                                sum_error += is_max * (*grad_next_layer)(i, j, z);
                                        }
                                }
                                (*input_gradients)(x, y, z) = sum_error;
                                gradientFrameBuffer->set(x, y, (int)sum_error);
                        }
                }
        }

        for(int i=0; i<input_size.depth; i++) {
                TensorRenderFrameBuffer* gradientFrameBuffer = gridRenderFrameBuffer->get(1, i);
                gradientFrameBuffer->swapBuffers();
        }

}

~PoolLayer() {
        delete gridRenderFrameBuffer;
        delete input_gradients;
        delete input;
        delete output;
        //TODO: Implement proper delete allocated filters

}

};

}

#endif
