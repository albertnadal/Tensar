#include <fstream>
#include <pthread.h>
#include <vector>
#include <mutex>

#include "src/common.cpp"
#include "src/tensor.cpp"
#include "src/tensor_float.cpp"
#include "src/tensor_gradient.cpp"
#include "src/gradient.cpp"
#include "src/layer.cpp"
#include "src/convolutional_layer.cpp"
#include "src/relu_layer.cpp"
#include "src/pool_layer.cpp"
#include "src/fully_connected_layer.cpp"
#include "src/input_case.cpp"
#include "src/tensor_render_frame_buffer.cpp"
#include "src/layer_grid_frame_buffer.cpp"

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#define WIDTH 1280
#define HEIGHT 740

using namespace std;

namespace NeuralNetwork {

class InputCase;
class Layer;

int timebase_timestamp = 0;
int frame_counter = 0;
char current_fps_buffer[20];
float avg_error_percent = 100.0f;
char avg_error_percent_buffer[30];
unsigned char *producer_frame_buffer = NULL;
unsigned char *consumer_frame_buffer = NULL;
mutex consumer_mutex;
bool is_consuming_frame_buffer = false;

vector<Layer*> layers;

class Network {
vector<Layer*> layers;

public:

Network() {

}

};


}

/***********************************/


using namespace NeuralNetwork;

uint32_t byteswap_uint32(uint32_t a)
{
        return ((((a >> 24) & 0xff) << 0) |
                (((a >> 16) & 0xff) << 8) |
                (((a >> 8) & 0xff) << 16) |
                (((a >> 0) & 0xff) << 24));
}

uint8_t* read_file( const char* szFile )
{
        ifstream file( szFile, ios::binary | ios::ate );
        streamsize size = file.tellg();
        file.seekg( 0, ios::beg );

        if ( size == -1 )
                return nullptr;

        uint8_t* buffer = new uint8_t[size];
        file.read( (char*)buffer, size );
        return buffer;
}

void drawString(int x, int y, char* msg, void *font = GLUT_BITMAP_HELVETICA_10) {
        glColor3d(0.0, 0.0, 0.0);
        glRasterPos2d(x, HEIGHT - y);
        for (const char *c = msg; *c != '\0'; c++) {
                glutBitmapCharacter(font, *c);
        }
}

GLuint LoadTextureWithTensorRenderFrameBuffer(TensorRenderFrameBuffer *tensorFrameBuffer)
{
        tensorFrameBuffer->consumer_mutex.lock();
        tensorFrameBuffer->is_consuming_frame_buffer = true;

        if(tensorFrameBuffer->consumer_frame_buffer == NULL) {
                return 0;
        }

        if(tensorFrameBuffer->texture) {
                // Delete previous generated texture
                glDeleteTextures(1, &tensorFrameBuffer->texture);
        }

        glGenTextures( 1, &tensorFrameBuffer->texture );
        glBindTexture( GL_TEXTURE_2D, tensorFrameBuffer->texture );
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGBA, tensorFrameBuffer->texture_width, tensorFrameBuffer->texture_height, GL_RGBA, GL_UNSIGNED_BYTE, tensorFrameBuffer->consumer_frame_buffer);

        tensorFrameBuffer->is_consuming_frame_buffer = false;
        tensorFrameBuffer->consumer_mutex.unlock();

        return tensorFrameBuffer->texture;
}

GLuint LoadTexture()
{
        if(consumer_frame_buffer == NULL) {
                return 0;
        }

        GLuint texture;
        int width = 28;
        int height = 28;

        glGenTextures( 1, &texture );
        glBindTexture( GL_TEXTURE_2D, texture );
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
        glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST );

        gluBuild2DMipmaps( GL_TEXTURE_2D, GL_RGB, width, height, GL_RGB, GL_UNSIGNED_BYTE, consumer_frame_buffer);

        return texture;
}

void display(void)
{
        frame_counter++;

        glClear(GL_COLOR_BUFFER_BIT);

        if(glutGet(GLUT_ELAPSED_TIME) - timebase_timestamp > 1000) {
                int current_timestamp = glutGet(GLUT_ELAPSED_TIME);
                snprintf(current_fps_buffer, 20, "%4.1ffps", (frame_counter * 1000.0)/(current_timestamp - timebase_timestamp));
                timebase_timestamp = current_timestamp;
                frame_counter = 0;
        }

        glColor3f(1,1,1);

        glEnable(GL_TEXTURE_2D);
        int x_offset = 20;
        int y_offset = 0;
        for(Layer* layer: layers) {
                LayerGridFrameBuffer *grid = layer->gridRenderFrameBuffer;
                x_offset += 50; // horizontal space between layers
                drawString(x_offset + 10, 25, grid->title, GLUT_BITMAP_HELVETICA_18);
                glColor3f(1,1,1);

                for(int x=0; x<grid->width; x++) {

                        x_offset += 10;
                        y_offset = 44;

                        drawString(x_offset, 50, grid->column_titles[x], GLUT_BITMAP_HELVETICA_10);
                        drawString(x_offset, 65, grid->column_subtitles[x], GLUT_BITMAP_TIMES_ROMAN_10);
                        glColor3f(1,1,1);

                        for(int y=0; y<grid->height; y++) {

                                y_offset += 24;
                                TensorRenderFrameBuffer* tensorFrameBuffer = grid->get(x, y);
                                GLuint texture = LoadTextureWithTensorRenderFrameBuffer(tensorFrameBuffer);
                                int tile_width = (tensorFrameBuffer->texture_width * 30) / tensorFrameBuffer->width;
                                int tile_height = (tensorFrameBuffer->texture_height * 30) / tensorFrameBuffer->height;

                                if(texture) {
                                        glBindTexture(GL_TEXTURE_2D, texture);
                                        glBegin(GL_QUADS);
                                        glTexCoord2f(0, 0);
                                        glVertex2f(x_offset, HEIGHT - y_offset);
                                        glTexCoord2f(0, 1);
                                        glVertex2f(x_offset, HEIGHT - y_offset - tile_height);
                                        glTexCoord2f(1, 1);
                                        glVertex2f(x_offset + tile_width, HEIGHT - y_offset - tile_height);
                                        glTexCoord2f(1, 0);
                                        glVertex2f(x_offset + tile_width, HEIGHT - y_offset);
                                        glEnd();
                                }

                                y_offset += 30;
                        }
                        x_offset += 30;
                }
        }
        glDisable(GL_TEXTURE_2D);

        drawString(WIDTH - 45, 15, current_fps_buffer, GLUT_BITMAP_HELVETICA_10);
        snprintf(avg_error_percent_buffer, 30, "%4.5f Avg. error", avg_error_percent);
        drawString(WIDTH - 220, HEIGHT - 18, avg_error_percent_buffer, GLUT_BITMAP_TIMES_ROMAN_24);

        glFlush();
        glutSwapBuffers();
}

vector<InputCase*> read_test_cases()
{
        vector<InputCase*> cases;

        producer_frame_buffer = (unsigned char *)malloc((28 * 28 * 3));
        consumer_frame_buffer = (unsigned char *)malloc((28 * 28 * 3));

        uint8_t* train_image = read_file( "train-images.idx3-ubyte" );
        uint8_t* train_labels = read_file( "train-labels.idx1-ubyte" );
        uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

        for(int i = 0; i < case_count; i++)
        {
                NeuralNetwork::size_tensor input_size{28, 28, 1};
                NeuralNetwork::size_tensor output_size{10, 1, 1};

                InputCase *c = new InputCase(input_size, output_size);

                uint8_t* img = train_image + 16 + i * (28 * 28);
                uint8_t* label = train_labels + 8 + i;

                for ( int x = 0; x < 28; x++ )
                        for ( int y = 0; y < 28; y++ ) {
                                (*c->data)(x, y, 0) = img[x + y * 28] / 255.f;
                                producer_frame_buffer[x * 3 + y * 28 * 3] = 0;
                                producer_frame_buffer[x * 3 + y * 28 * 3 + 1] = img[x + y * 28]; // green byte
                                producer_frame_buffer[x * 3 + y * 28 * 3 + 2] = 0;
                        }

                if(!is_consuming_frame_buffer) {
                        consumer_mutex.lock();
                        unsigned char *tmp = producer_frame_buffer;
                        producer_frame_buffer = consumer_frame_buffer;
                        consumer_frame_buffer = tmp;
                        consumer_mutex.unlock();
                }

                for ( int b = 0; b < 10; b++ ) {
                        (*c->output)(b, 0, 0) = *label == b ? 1.0f : 0.0f;
                }

                cases.push_back(c);
        }

        delete[] train_image;
        delete[] train_labels;
        return cases;
}

void idle(void)
{
        glutPostRedisplay();
}

void reshape(int w, int h)
{
        glViewport(0, 0, WIDTH, HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, (GLdouble) w, 0.0, (GLdouble) h);
}

float train(vector<Layer*> &layers, InputCase *input_case)//TensorFloat *data, TensorFloat *expected)
{
        for(int i = 0; i < layers.size(); i++) {
                Layer *layer = layers[i];

                if(i == 0) { layer->activate(input_case->data); }
                else       { layer->activate(layers[i - 1]->output); }
        }

        //output of the last layer must have the same size as the case expected size
        TensorFloat* diff_gradient = TensorFloat::diff(layers.back()->output, input_case->output); // difference between the neural network output and expected output

        for(int i = layers.size() - 1; i >= 0; i--) {
                if(i == layers.size() - 1)  { layers[i]->calc_grads(diff_gradient); }
                else                        { layers[i]->calc_grads(layers[i + 1]->input_gradients); }
        }

        for(int i = 0; i < layers.size(); i++) {
                layers[i]->fix_weights();
        }

        float err = 0;

        //check if the output of the last layer have the same size as the case expected size
        if((diff_gradient->size.width == input_case->output->size.width) && (diff_gradient->size.height == input_case->output->size.height) && (diff_gradient->size.depth == input_case->output->size.depth)) {
                //calculate the error %
                for(int i = 0; i < diff_gradient->size.width * diff_gradient->size.height * diff_gradient->size.depth; i++) {
                        float f = input_case->output->values[i];
                        if(f > 0.5)
                                err += abs(diff_gradient->values[i]);
                }
        }

        delete diff_gradient;
        return err * 100;
}

static void* tensarThreadFunc(void* v) {
        vector<InputCase*> cases = read_test_cases();

        ConvolutionalLayer *cnn_layer1 = new ConvolutionalLayer(1, 5, 8, cases[0]->data->size); // 28 * 28 * 1 -> 24 * 24 * 8
        ReLuLayer *relu_layer1 = new ReLuLayer(cnn_layer1->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        PoolLayer *pool_layer1 = new PoolLayer(2, 2, relu_layer1->output->size);
/*
        ConvolutionalLayer *cnn_layer2 = new ConvolutionalLayer(1, 3, 10, pool_layer1->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        ReLuLayer *relu_layer2 = new ReLuLayer(cnn_layer2->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        PoolLayer *pool_layer2 = new PoolLayer(2, 2, relu_layer2->output->size);
 */
        FullyConnectedLayer *fc_layer = new FullyConnectedLayer(pool_layer1->output->size, {10, 1, 1});

        layers.push_back(cnn_layer1);
        layers.push_back(relu_layer1);
        layers.push_back(pool_layer1);
/*
        layers.push_back(cnn_layer2);
        layers.push_back(relu_layer2);
        layers.push_back(pool_layer2);
 */
        layers.push_back(fc_layer);

        float amse = 0;
        int ic = 0;

        cout << "Start training...\n";

        for(long ep = 0; ep < 100000;)
        {
                for(int i=0; i<cases.size(); i++)
                {
                        InputCase *input_case = cases[i];
                        float xerr = train(layers, input_case);
                        amse += xerr;

                        ep++;
                        ic++;
                        avg_error_percent = amse/ic;

                        if(ep % 1000 == 0) {
                                cout << "case " << ep << " err=" << avg_error_percent << endl;

                                TensorFloat* expected = input_case->output;
                                cout << "Expected:\n";
                                for(int e = 0; e < 10; e++) {
                                        printf("[%i] %f\n", e, (*expected)(e, 0, 0)*100.0f);
                                }

                                cout << "Output:\n";
                                TensorFloat* output = layers.back()->output;
                                for(int o = 0; o < 10; o++) {
                                        printf("[%i] %f\n", o, (*output)(o, 0, 0)*100.0f);
                                }
                        }
                }
        }
        return 0;
}

int main(int argc, char *argv[]) {

        pthread_t tensarThreadId;
        pthread_create(&tensarThreadId, NULL, tensarThreadFunc, 0);

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB);
        glutInitWindowSize(WIDTH, HEIGHT);
        glutInitWindowPosition(0, 0);
        glutCreateWindow("Tensar");
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutIdleFunc(idle);

        glClearColor(1.0, 1.0, 1.0, 1.0);
        //glEnable(GL_LINE_SMOOTH);

        glutMainLoop();

        return 0;
}
