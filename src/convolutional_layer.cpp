#ifndef _CONVOLUTIONAL_LAYER_CPP
#define _CONVOLUTIONAL_LAYER_CPP

#include <cassert>
#include <vector>
#include <cmath>
#include "layer.cpp"
#include "tensor_gradient.cpp"
#include "tensor_float.cpp"
#include "layer_grid_frame_buffer.cpp"
#include "tensor_render_frame_buffer.cpp"

namespace NeuralNetwork {

class ConvolutionalLayer : public Layer {

public:

vector<TensorFloat*> filters;
vector<TensorGradient*> filter_gradients;
int stride, extend_filter;

ConvolutionalLayer(int stride, int extend_filter, int number_filters, size_tensor in_size) {
        type = LayerType::convolutional;
        input_size = in_size;

        //Define and alloc a grid layout for render buffers

        // {input, filter, gradient, output}
        // {cellObj, cellObj, cellObj, cellObj}
        // {cellObj, cellObj, cellObj, cellObj}
        // ...
        // {cellObj, cellObj, cellObj, cellObj}

        gridRenderFrameBuffer = new LayerGridFrameBuffer(3, number_filters, (char*)"Convolutional"); // 3 = {input, filter, gradient, output}

        // Initialize first column of the grid with Inputs buffers
        gridRenderFrameBuffer->column_titles.push_back((char*)"in");
        char *subtitle = new char[50];
        sprintf(subtitle, "%d x %d", in_size.width, in_size.height);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<number_filters; i++) {
                gridRenderFrameBuffer->set(0, i, new TensorRenderFrameBuffer(in_size.width, in_size.height));
        }

        // Initialize second column of the grid with Filters buffers
        gridRenderFrameBuffer->column_titles.push_back((char*)"filter");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", extend_filter, extend_filter);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<number_filters; i++) {
                gridRenderFrameBuffer->set(1, i, new TensorRenderFrameBuffer(extend_filter, extend_filter));
        }

        // Initialize forth column of the grid with Output buffers
        gridRenderFrameBuffer->column_titles.push_back((char*)"out");
        subtitle = new char[50];
        sprintf(subtitle, "%d x %d", (in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1);
        gridRenderFrameBuffer->column_subtitles.push_back(subtitle);

        for(int i=0; i<number_filters; i++) {
                gridRenderFrameBuffer->set(2, i, new TensorRenderFrameBuffer((in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1));
        }

        input_gradients = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        input = new TensorFloat(in_size.width, in_size.height, in_size.depth);
        output = new TensorFloat((in_size.width - extend_filter) / stride + 1, (in_size.height - extend_filter) / stride + 1, number_filters);
        this->stride = stride;
        this->extend_filter = extend_filter;
        assert( (float( in_size.width - extend_filter ) / stride + 1) == ((in_size.width - extend_filter) / stride + 1) );
        assert( (float( in_size.height - extend_filter ) / stride + 1) == ((in_size.height - extend_filter) / stride + 1) );

        TensorRenderFrameBuffer* filterFrameBuffer;

        for(int a = 0; a < number_filters; a++) {
                filterFrameBuffer = gridRenderFrameBuffer->get(1, a);
                TensorFloat *filter = new TensorFloat(extend_filter, extend_filter, in_size.depth);
                int maxval = extend_filter * extend_filter * in_size.depth;

                for(int x = 0; x < extend_filter; x++)
                {
                        for(int y = 0; y < extend_filter; y++)
                        {
                                for(int z = 0; z < in_size.depth; z++)
                                {
                                        float value = 1.0f / maxval * rand() / float( RAND_MAX );
                                        (*filter)(x, y, z) = value;
                                        filterFrameBuffer->set(x, y, (int)(value * 255));
                                }
                        }
                }

                filters.push_back(filter);
                filterFrameBuffer->swapBuffers();
        }

        for(int i = 0; i < number_filters; i++) {
                TensorGradient *tensorGradient = new TensorGradient(extend_filter, extend_filter, in_size.depth);

                // Update render frame gradients buffer values
                for(int x = 0; x < extend_filter; x++)
                {
                        for(int y = 0; y < extend_filter; y++)
                        {
                                for(int z = 0; z < in_size.depth; z++)
                                {
                                        Gradient *gradient = tensorGradient->get(x, y, z);
                                        float value = gradient->grad;
                                }
                        }
                }

                filter_gradients.push_back(tensorGradient);
        }

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
                       (int)filters.size() - 1,
        };
}

void activate(TensorFloat *in) {
        this->input = in;

        // Update render frame inputs buffer values
        for(int filter = 0; filter < filters.size(); filter++)
        {
                TensorRenderFrameBuffer* inputFrameBuffer = gridRenderFrameBuffer->get(0, filter);
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
        }

        activate();
}

void activate() {

        for(int filter = 0; filter < filters.size(); filter++)
        {
                TensorRenderFrameBuffer* outputFrameBuffer = gridRenderFrameBuffer->get(2, filter);
                TensorFloat *filter_data = filters[filter];
                for(int y = 0; y < output->size.height; y++)
                {
                        for(int x = 0; x < output->size.width; x++)
                        {
                                point_tensor mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
                                float sum = 0;
                                for(int i = 0; i < extend_filter; i++)
                                {
                                        for(int j = 0; j < extend_filter; j++)
                                        {
                                                for(int z = 0; z < input->size.depth; z++)
                                                {
                                                        float f = (*filter_data)( i, j, z );
                                                        float v = (*input)( mapped.x + i, mapped.y + j, z );
                                                        sum += f*v;
                                                }
                                        }
                                }
                                (*output)(x, y, filter) = sum;
                                outputFrameBuffer->set(x, y, (int)(sum * 255));
                        }
                }
                outputFrameBuffer->swapBuffers();
        }

}

void fix_weights() {

        TensorRenderFrameBuffer* filterFrameBuffer;
        for(int k = 0; k < filters.size(); k++)
        {
                filterFrameBuffer = gridRenderFrameBuffer->get(1, k);
                for(int y = 0; y < extend_filter; y++)
                {
                        for(int x = 0; x < extend_filter; x++)
                        {
                                for(int z = 0; z < input->size.depth; z++)
                                {
                                        TensorFloat *filter = filters[k];
                                        float& w = filter->get(x, y, z);
                                        TensorGradient *tensor_gradient = filter_gradients[k];
                                        Gradient *grad = tensor_gradient->get(x, y, z);
                                        w = update_weight(w, grad);
                                        update_gradient(grad);
                                        filterFrameBuffer->set128(x, y, (int)((w * 128)/0.5f)); // signed value between -128 and 128
                                }
                        }
                }
                filterFrameBuffer->swapBuffers();
        }

}

void calc_grads(TensorFloat* grad_next_layer) {

        // Reset all layer gradients to 0
        for (int k = 0; k < filter_gradients.size(); k++) {
                TensorGradient *gradient = filter_gradients[k]; //gradient->get(x, y, z).grad;

                for ( int x = 0; x < extend_filter; x++ ) {
                        for ( int y = 0; y < extend_filter; y++ ) {
                                for ( int z = 0; z < input->size.depth; z++ ) {
                                        gradient->get(x, y, z)->grad = 0;
                                }
                        }
                }
        }

        for(int x = 0; x < input->size.width; x++) {
                for(int y = 0; y < input->size.height; y++) {
                        range_tensor rn = map_to_output(x, y);
                        for(int z = 0; z < input->size.depth; z++) {
                                float sum_error = 0;
                                for(int i = rn.min_x; i <= rn.max_x; i++) {
                                        int minx = i * stride;
                                        for(int j = rn.min_y; j <= rn.max_y; j++) {
                                                int miny = j * stride;
                                                for(int k = 0; k < filters.size(); k++) {
                                                        TensorGradient *tensorGradient = filter_gradients[k];
                                                        TensorFloat *tensorFilter = filters[k];
                                                        int w_applied = tensorFilter->get( x - minx, y - miny, z );
                                                        sum_error += w_applied * (*grad_next_layer)( i, j, k );
                                                        float value = (*input)( x, y, z ) * (*grad_next_layer)( i, j, k );

                                                        Gradient *gradient = tensorGradient->get(x - minx, y - miny, z);
                                                        gradient->grad += value;
                                                }
                                        }
                                }
                                (*input_gradients)(x, y, z) = sum_error;
                        }
                }
        }

}

~ConvolutionalLayer() {
        for(int f=0; f<filters.size(); f++)
                delete filters[f];

        for(int i=0; i<filter_gradients.size(); i++)
                delete filter_gradients[i];
        delete input_gradients;
        delete input;
        delete output;
        delete gridRenderFrameBuffer;
}

};

}

#endif
