#ifndef _LAYER_GRID_FRAME_BUFFER_CPP
#define _LAYER_GRID_FRAME_BUFFER_CPP

#include <cassert>
#include <vector>
#include "tensor_render_frame_buffer.cpp"

using namespace std;

namespace NeuralNetwork {

class LayerGridFrameBuffer {

public:

int width;
int height;
char title[100] = "\0";
vector<char*> column_titles;
vector<char*> column_subtitles;
TensorRenderFrameBuffer **cells = NULL;

LayerGridFrameBuffer() {

}

LayerGridFrameBuffer(int _width, int _height, char *_title) {
        strcpy(title, _title);

        // Creates an array of pointers
        cells = new TensorRenderFrameBuffer*[_width * _height];

        // Initialize the array with null pointers
        for(int i=0; i<_width * _height; i++)
                cells[i] = NULL;
        width = _width;
        height = _height;
}

void setTitleForColumn(int col_index) {

}

TensorRenderFrameBuffer* operator()(int x, int y) const
{
        return this->get(x, y);
}

TensorRenderFrameBuffer* get(int x, int y) const
{
        assert(x >= 0 && y >= 0);
        assert(x < width && y < height);
        return cells[(y * width) + x];
}

void set(int x, int y, TensorRenderFrameBuffer* cell) const
{
        assert(x >= 0 && y >= 0);
        assert(x < width && y < height);
        cells[(y * width) + x] = cell;
}

~LayerGridFrameBuffer() {
        if(cells != NULL) {
                for(int i=0; i<width*height; i++)
                        if(cells[i] != NULL)
                                delete cells[i];
                delete[] cells;
        }

        for(int i=0; i<column_titles.size(); i++)
                delete column_titles[i];

        for(int i=0; i<column_subtitles.size(); i++)
                delete column_subtitles[i];

}

};

}

#endif
