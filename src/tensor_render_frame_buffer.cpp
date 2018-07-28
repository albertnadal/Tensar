#ifndef _TENSOR_RENDER_FRAME_BUFFER_CPP
#define _TENSOR_RENDER_FRAME_BUFFER_CPP

#include <cassert>
#include <mutex>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

using namespace std;

namespace NeuralNetwork {

class TensorRenderFrameBuffer {

public:

int width; // pixels
int height; // pixels
int texture_width; // pixels
int texture_height; // pixels
char *caption = NULL;
GLuint texture = 0;

unsigned char *producer_frame_buffer = NULL;
unsigned char *consumer_frame_buffer = NULL;
mutex consumer_mutex;
bool is_consuming_frame_buffer = false;

TensorRenderFrameBuffer(int _width, int _height) {
        texture_width = 64;
        texture_height = 64;
        width = _width;
        height = _height;

        producer_frame_buffer = (unsigned char *)malloc((texture_width * texture_height * 4));
        consumer_frame_buffer = (unsigned char *)malloc((texture_width * texture_height * 4));

        for(int x=0; x<texture_width; x++) {
                for(int y=0; y<texture_height; y++) {
                        producer_frame_buffer[(y * texture_width * 4) + x * 4] = 255;
                        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 1] = 255; // green
                        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 2] = 255;
                        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 3] = 0; //Alpha

                        consumer_frame_buffer[(y * texture_width * 4) + x * 4] = 255;
                        consumer_frame_buffer[(y * texture_width * 4) + x * 4 + 1] = 255; // green
                        consumer_frame_buffer[(y * texture_width * 4) + x * 4 + 2] = 255;
                        consumer_frame_buffer[(y * texture_width * 4) + x * 4 + 3] = 0; //Alpha
                }
        }
}

void set(int x, int y, signed int value) const
{
        assert(x >= 0 && y >= 0);
        assert(x < texture_width && y < texture_height);
        producer_frame_buffer[(y * texture_width * 4) + x * 4] = (value < 0) ? (unsigned char)abs(value) : 0; // red
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 1] = (value >= 0) ? (unsigned char)value : 0; // green
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 2] = 0;
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 3] = 1.0f; //Alpha
}

void set128(int x, int y, int value) const
{
        assert(x >= 0 && y >= 0);
        assert(x < texture_width && y < texture_height);
        producer_frame_buffer[(y * texture_width * 4) + x * 4] = (value < 0) ? (unsigned char)(abs(value) * 2) : 0; // red
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 1] = (value >= 0) ? (unsigned char)(value * 2) : 0; // green
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 2] = 0;
        producer_frame_buffer[(y * texture_width * 4) + x * 4 + 3] = 1.0f; //Alpha
}

signed int get(int x, int y) const
{
        assert(x >= 0 && y >= 0);
        assert(x < texture_width && y < texture_height);
        signed int negative_value = consumer_frame_buffer[(y * texture_width * 4) + x * 4] * -1.0f;
        signed int positive_valye = consumer_frame_buffer[(y * texture_width * 4) + x * 4 + 1];
        return negative_value + positive_valye;
}

void swapBuffers()
{
        if(!is_consuming_frame_buffer) {
                consumer_mutex.lock();
                unsigned char *tmp = producer_frame_buffer;
                producer_frame_buffer = consumer_frame_buffer;
                consumer_frame_buffer = tmp;
                consumer_mutex.unlock();
        }
}

~TensorRenderFrameBuffer() {

        if(producer_frame_buffer != NULL) {
                delete[] producer_frame_buffer;
        }

        if(consumer_frame_buffer != NULL) {
                delete[] consumer_frame_buffer;
        }
}

};

}

#endif
