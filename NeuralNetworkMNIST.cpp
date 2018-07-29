#include <fstream>
#include <pthread.h>
#include <vector>
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

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_DEPTH 1

#define OUTPUT_WIDTH 10
#define OUTPUT_HEIGHT 1
#define OUTPUT_DEPTH 1

#define SCREEN_WIDTH 1280
#define SCREEN_HEIGHT 740

using namespace std;
using namespace NeuralNetwork;

vector<Layer*> layers;
bool paused = false;
TensorRenderFrameBuffer* currentInputTensorFrameBuffer = NULL;
TensorRenderFrameBuffer* selectedTensorFrameBuffer = NULL;
int iteration = 0;
int expected_label, predicted_label;
int timebase_timestamp = 0;
int frame_counter = 0;
float avg_error_percent = 100.0f;
char current_fps_buffer_msg[20];
char expected_output_buffer_msg[60];
char avg_error_percent_buffer_msg[30];
char press_space_buffer_msg[40];
int mouse_x = 0;
int mouse_y = 0;

uint32_t byteswapUint32(uint32_t a)
{
        return ((((a >> 24) & 0xff) << 0) |
                (((a >> 16) & 0xff) << 8) |
                (((a >> 8) & 0xff) << 16) |
                (((a >> 0) & 0xff) << 24));
}

uint8_t* readFile( const char* szFile )
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
        glRasterPos2d(x, SCREEN_HEIGHT - y);
        for (const char *c = msg; *c != '\0'; c++) {
                glutBitmapCharacter(font, *c);
        }
}

GLuint loadTextureWithTensorRenderFrameBuffer(TensorRenderFrameBuffer *tensorFrameBuffer)
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

void display(void)
{
        frame_counter++;

        /*** BEGIN: 2D tensors ***/
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, (GLdouble) SCREEN_WIDTH, 0.0, (GLdouble) SCREEN_HEIGHT);

        glClear(GL_COLOR_BUFFER_BIT);

        if(glutGet(GLUT_ELAPSED_TIME) - timebase_timestamp > 1000) {
                int current_timestamp = glutGet(GLUT_ELAPSED_TIME);
                snprintf(current_fps_buffer_msg, 20, "%4.1ffps", (frame_counter * 1000.0)/(current_timestamp - timebase_timestamp));
                timebase_timestamp = current_timestamp;
                frame_counter = 0;
        }

        glColor3f(1,1,1);

        glEnable(GL_TEXTURE_2D);
        int x_offset = 20;
        int y_offset = 0;

        // Draw the current input case
        if(currentInputTensorFrameBuffer != NULL) {
                drawString(x_offset, 25, (char*)"Input", GLUT_BITMAP_HELVETICA_18);
                y_offset = 44;
                drawString(x_offset, 50, (char*)"in", GLUT_BITMAP_HELVETICA_10);
                drawString(x_offset, 65, (char*)"28 x 28", GLUT_BITMAP_TIMES_ROMAN_10);
                y_offset += 24;

                GLuint texture = loadTextureWithTensorRenderFrameBuffer(currentInputTensorFrameBuffer);
                int tile_width = (currentInputTensorFrameBuffer->texture_width * 60) / currentInputTensorFrameBuffer->width;
                int tile_height = (currentInputTensorFrameBuffer->texture_height * 60) / currentInputTensorFrameBuffer->height;
                if(texture) {
                        glColor3f(1,1,1);
                        glBindTexture(GL_TEXTURE_2D, texture);
                        glBegin(GL_QUADS);
                        glTexCoord2f(0, 0);
                        glVertex2f(x_offset, SCREEN_HEIGHT - y_offset);
                        glTexCoord2f(0, 1);
                        glVertex2f(x_offset, SCREEN_HEIGHT - y_offset - tile_height);
                        glTexCoord2f(1, 1);
                        glVertex2f(x_offset + tile_width, SCREEN_HEIGHT - y_offset - tile_height);
                        glTexCoord2f(1, 0);
                        glVertex2f(x_offset + tile_width, SCREEN_HEIGHT - y_offset);
                        glEnd();

                        if((x_offset <= mouse_x) && (mouse_x <= x_offset + 60) && (y_offset <= mouse_y) && (mouse_y <= y_offset + 60)) {
                                selectedTensorFrameBuffer = currentInputTensorFrameBuffer;
                                mouse_x = mouse_y = 0;
                        }
                }
                x_offset += 40;
        }

        // Draw all the neural network layers
        for(Layer* layer: layers) {
                LayerGridFrameBuffer *grid = layer->gridRenderFrameBuffer;
                x_offset += 50; // horizontal space between layers
                drawString(x_offset + 20, 25, grid->title, GLUT_BITMAP_HELVETICA_18);
                glColor3f(1,1,1);

                for(int x=0; x<grid->width; x++) {

                        x_offset += 24;
                        y_offset = 44;

                        drawString(x_offset, 50, grid->column_titles[x], GLUT_BITMAP_HELVETICA_10);
                        drawString(x_offset, 65, grid->column_subtitles[x], GLUT_BITMAP_TIMES_ROMAN_10);
                        glColor3f(1,1,1);

                        for(int y=0; y<grid->height; y++) {

                                y_offset += 24;
                                TensorRenderFrameBuffer* tensorFrameBuffer = grid->get(x, y);
                                GLuint texture = loadTextureWithTensorRenderFrameBuffer(tensorFrameBuffer);
                                int tile_width = (tensorFrameBuffer->texture_width * 40) / tensorFrameBuffer->width;
                                int tile_height = (tensorFrameBuffer->texture_height * 40) / tensorFrameBuffer->height;

                                if(texture) {
                                        glBindTexture(GL_TEXTURE_2D, texture);
                                        glBegin(GL_QUADS);
                                        glTexCoord2f(0, 0);
                                        glVertex2f(x_offset, SCREEN_HEIGHT - y_offset);
                                        glTexCoord2f(0, 1);
                                        glVertex2f(x_offset, SCREEN_HEIGHT - y_offset - tile_height);
                                        glTexCoord2f(1, 1);
                                        glVertex2f(x_offset + tile_width, SCREEN_HEIGHT - y_offset - tile_height);
                                        glTexCoord2f(1, 0);
                                        glVertex2f(x_offset + tile_width, SCREEN_HEIGHT - y_offset);
                                        glEnd();

                                        if((x_offset <= mouse_x) && (mouse_x <= x_offset + 40) && (y_offset <= mouse_y) && (mouse_y <= y_offset + 40)) {
                                                selectedTensorFrameBuffer = tensorFrameBuffer;
                                                mouse_x = mouse_y = 0;
                                        }
                                }

                                y_offset += 30;
                        }
                        x_offset += 30;
                }
        }
        glDisable(GL_TEXTURE_2D);

        if(paused) {
          drawString(20, SCREEN_HEIGHT - 100, (char*)"TRAINING PAUSED", GLUT_BITMAP_HELVETICA_18);
        }
        drawString(SCREEN_WIDTH - 45, 15, current_fps_buffer_msg, GLUT_BITMAP_HELVETICA_10);
        snprintf(expected_output_buffer_msg, 60, "Input #%d - Expected: %d - Predicted: %d [ %s ]", iteration, expected_label, predicted_label, (expected_label == predicted_label) ? (char*)"SUCCESS" : (char*)"FAIL");
        drawString(20, SCREEN_HEIGHT - 70, expected_output_buffer_msg, GLUT_BITMAP_9_BY_15);
        snprintf(avg_error_percent_buffer_msg, 30, "%4.5f Avg. error", avg_error_percent);
        drawString(20, SCREEN_HEIGHT - 45, avg_error_percent_buffer_msg, GLUT_BITMAP_9_BY_15);
        snprintf(press_space_buffer_msg, 40, "Press space to pause/continue training");
        drawString(20, SCREEN_HEIGHT - 20, press_space_buffer_msg, GLUT_BITMAP_8_BY_13);
        /*** END: 2D tensors ***/

        /*** BEGIN: 3D selected tensor chart ***/
        glLoadIdentity();
        glOrtho(0.0, SCREEN_WIDTH, 0.0, SCREEN_HEIGHT, 0.0, 1000.0f);
        glDepthFunc(GL_LESS);
        glDepthRange(0.0f, 1.0f);

        float axis_x_offset = 900.0f;
        float axis_y_offset = 250.0f;
        glRotatef(12.0f, -1.0f, 1.0f, 0.0f);

        glColor4ub(0, 0, 0, 255);
        glBegin(GL_LINE_STRIP);
        glVertex3f(0.0f + axis_x_offset, 0.0f + axis_y_offset, 0.0f);
        glVertex3f(300.0f + axis_x_offset, 0.0f + axis_y_offset, 0.0f);
        glEnd();
        glBegin(GL_LINE_STRIP);
        glVertex3f(0.0f + axis_x_offset, 0.0f + axis_y_offset, 0.0f);
        glVertex3f(0.0f + axis_x_offset, 100.0f + axis_y_offset, 0.0f);
        glEnd();
        glBegin(GL_LINE_STRIP);
        glVertex3f(0.0f + axis_x_offset, 0.0f + axis_y_offset, 0.0f);
        glVertex3f(0.0f + axis_x_offset, 0.0f + axis_y_offset, -980.0f);
        glEnd();

        glColor4ub(200, 55, 100, 255);
        glRasterPos3f(310.0f + axis_x_offset, 0.0f + axis_y_offset, 0.0f);
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'x');
        glRasterPos3f(0.0f + axis_x_offset, 110.0f + axis_y_offset, 0.0f);
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'y');
        glRasterPos3f(-5.0f + axis_x_offset, 15.0f + axis_y_offset, -840.0f);
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, 'z');

        if(selectedTensorFrameBuffer != NULL) {
                selectedTensorFrameBuffer->consumer_mutex.lock();
                selectedTensorFrameBuffer->is_consuming_frame_buffer = true;

                float cell_width = 300.0f / selectedTensorFrameBuffer->width;
                float cell_height = 900.0f / selectedTensorFrameBuffer->height;
                glMatrixMode(GL_MODELVIEW);
                glBegin(GL_QUADS);
                int i, _i, j, _j;
                int frameBufferWidth = max(selectedTensorFrameBuffer->width - 1, 1);
                int frameBufferHeight = max(selectedTensorFrameBuffer->height - 1, 1);
                for(i = 0; i < frameBufferWidth; _i = i++) {
                        for(j = 0; j < frameBufferHeight; j++) {
                                _i = (i == selectedTensorFrameBuffer->width - 1) ? i : i+1;
                                _j = (j == selectedTensorFrameBuffer->height - 1) ? j : j+1;

                                //top left
                                glColor4ub(selectedTensorFrameBuffer->getRed(i, j), selectedTensorFrameBuffer->getGreen(i, j), selectedTensorFrameBuffer->getBlue(i, j), 255);
                                glVertex3f (i*cell_width + axis_x_offset, ((selectedTensorFrameBuffer->getValue(i, j) * 50) / 255) + axis_y_offset, -j*cell_height);
                                //bottom left
                                glColor4ub(selectedTensorFrameBuffer->getRed(i, _j), selectedTensorFrameBuffer->getGreen(i, _j), selectedTensorFrameBuffer->getBlue(i, _j), 255);
                                glVertex3f (i*cell_width + axis_x_offset, ((selectedTensorFrameBuffer->getValue(i, _j) * 50) / 255) + axis_y_offset, -(j+1)*cell_height);
                                //bottom right
                                glColor4ub(selectedTensorFrameBuffer->getRed(_i, _j), selectedTensorFrameBuffer->getGreen(_i, _j), selectedTensorFrameBuffer->getBlue(_i, _j), 255);
                                glVertex3f ((i+1)*cell_width + axis_x_offset, ((selectedTensorFrameBuffer->getValue(_i, _j) * 50) / 255) + axis_y_offset, -(j+1)*cell_height);
                                //top right
                                glColor4ub(selectedTensorFrameBuffer->getRed(_i, j), selectedTensorFrameBuffer->getGreen(_i, j), selectedTensorFrameBuffer->getBlue(_i, j), 255);
                                glVertex3f ((i+1)*cell_width + axis_x_offset, ((selectedTensorFrameBuffer->getValue(_i, j) * 50) / 255) + axis_y_offset, -(j)*cell_height);
                        }
                }
                glEnd();

                selectedTensorFrameBuffer->is_consuming_frame_buffer = false;
                selectedTensorFrameBuffer->consumer_mutex.unlock();
        }
        /*** END: 3D selected tensor chart ***/

        glFlush();
        glutSwapBuffers();
}

vector<InputCase*> readInputDataset()
{
        vector<InputCase*> cases;

        uint8_t* train_image = readFile( "train-images.idx3-ubyte" );
        uint8_t* train_labels = readFile( "train-labels.idx1-ubyte" );
        uint32_t case_count = byteswapUint32( *(uint32_t*)(train_image + 4) );

        for(int i = 0; i < case_count; i++)
        {
                size_tensor input_size{INPUT_WIDTH, INPUT_HEIGHT, INPUT_DEPTH};
                size_tensor output_size{OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DEPTH};

                InputCase *c = new InputCase(input_size, output_size);

                uint8_t* img = train_image + 16 + i * (INPUT_WIDTH * INPUT_HEIGHT);
                uint8_t* label = train_labels + 8 + i;

                for ( int x = 0; x < INPUT_WIDTH; x++ )
                        for ( int y = 0; y < INPUT_HEIGHT; y++ ) {
                                (*c->data)(x, y, 0) = img[x + y * INPUT_WIDTH] / 255.f;
                        }

                for ( int b = 0; b < OUTPUT_WIDTH; b++ ) {
                        (*c->output)(b, 0, 0) = *label == b ? 1.0f : 0.0f;
                }

                cases.push_back(c);
        }

        delete[] train_image;
        delete[] train_labels;
        return cases;
}

static void keyboard(int key, int x, int y) {
        switch (key) {
        case GLUT_KEY_LEFT:
                break;
        case GLUT_KEY_RIGHT:
                break;
        case GLUT_KEY_UP:
                break;
        case GLUT_KEY_DOWN:
                break;
        case 32: // space bar
                paused = !paused;
                break;
        }
}

static void mouse(int button, int state, int x, int y) {
        mouse_x = x;
        mouse_y = y;
}

void idle(void)
{
        glutPostRedisplay();
}

void reshape(int w, int h)
{
        glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
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
        vector<InputCase*> cases = readInputDataset(); // MNIST dataset
        currentInputTensorFrameBuffer = new TensorRenderFrameBuffer(INPUT_WIDTH, INPUT_HEIGHT); // frame buffer for rendering the current input tensor from MNIST dataset

        /*** BEGIN: Simple Convolutional Neural Network topology model ***/
        ConvolutionalLayer *cnn_layer1 = new ConvolutionalLayer(1, 5, 8, cases[0]->data->size); // 28 * 28 * 1 -> 24 * 24 * 8
        ReLuLayer *relu_layer1 = new ReLuLayer(cnn_layer1->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        PoolLayer *pool_layer1 = new PoolLayer(2, 2, relu_layer1->output->size);
        FullyConnectedLayer *fc_layer = new FullyConnectedLayer(pool_layer1->output->size, {OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DEPTH});

        layers.push_back(cnn_layer1);
        layers.push_back(relu_layer1);
        layers.push_back(pool_layer1);
        layers.push_back(fc_layer);
        /*** END: Simple Convolutional Neural Network topology model ***/

        /*** BEGIN: Yet another Convolutional Neural Network topology model ***/
/*
        ConvolutionalLayer *cnn_layer1 = new ConvolutionalLayer(1, 5, 8, cases[0]->data->size); // 28 * 28 * 1 -> 24 * 24 * 8
        ReLuLayer *relu_layer1 = new ReLuLayer(cnn_layer1->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        PoolLayer *pool_layer1 = new PoolLayer(2, 2, relu_layer1->output->size);
        ConvolutionalLayer *cnn_layer2 = new ConvolutionalLayer(1, 3, 10, pool_layer1->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        ReLuLayer *relu_layer2 = new ReLuLayer(cnn_layer2->output->size); // 28 * 28 * 1 -> 24 * 24 * 8
        PoolLayer *pool_layer2 = new PoolLayer(2, 2, relu_layer2->output->size);
        FullyConnectedLayer *fc_layer = new FullyConnectedLayer(pool_layer2->output->size, {OUTPUT_WIDTH, OUTPUT_HEIGHT, OUTPUT_DEPTH});

        layers.push_back(cnn_layer1);
        layers.push_back(relu_layer1);
        layers.push_back(pool_layer1);
        layers.push_back(cnn_layer2);
        layers.push_back(relu_layer2);
        layers.push_back(pool_layer2);
        layers.push_back(fc_layer);
 */
        /*** END: Yet another Convolutional Neural Network topology model ***/

        float amse = 0;
        float max_value = 0.0f;
        TensorFloat* expected;
        TensorFloat* output;

        cout << "Start training...\n";
        for(long ep = 0; ep < 100000;)
        {
                for(int i=0; i<cases.size(); i++)
                {
                        while(1) {
                          cout << "";
                          if(!paused)
                            break;
                        }

                        InputCase *input_case = cases[i];

                        // update the frame buffer with the current input values
                        for(int x = 0; x < input_case->data->size.width; x++)
                                for(int y = 0; y < input_case->data->size.height; y++)
                                        for(int z = 0; z < input_case->data->size.depth; z++)
                                        {
                                                float value = input_case->data->get(x, y, z);
                                                currentInputTensorFrameBuffer->set(x, y, (int)(value * 255));
                                        }
                        // render input case swapping the double buffers
                        currentInputTensorFrameBuffer->swapBuffers();

                        // train the layers with the current input case
                        float xerr = train(layers, input_case);

                        // Calculate the average error of the training
                        amse += xerr;
                        ep++;
                        iteration++;
                        avg_error_percent = amse/iteration;

                        expected = input_case->output;
                        max_value = 0.0f;
                        for(int e = 0; e < 10; e++)
                          if(max_value < (*expected)(e, 0, 0)) {
                            max_value = (*expected)(e, 0, 0);
                            expected_label = e;
                          }

                        output = layers.back()->output;
                        max_value = 0.0f;
                        for(int o = 0; o < 10; o++)
                          if(max_value < (*output)(o, 0, 0)) {
                            max_value = (*output)(o, 0, 0);
                            predicted_label = o;
                          }

                        if(ep % 1000 == 0) {
                                cout << "case " << ep << " err=" << avg_error_percent << endl;

                                expected = input_case->output;
                                cout << "Expected:\n";
                                for(int e = 0; e < 10; e++) {
                                        printf("[%i] %f\n", e, (*expected)(e, 0, 0)*100.0f);
                                }

                                cout << "Output:\n";
                                output = layers.back()->output;
                                for(int o = 0; o < 10; o++) {
                                        printf("[%i] %f\n", o, (*output)(o, 0, 0)*100.0f);
                                }
                        }
                }
        }
        delete currentInputTensorFrameBuffer;
        return 0;
}

int main(int argc, char *argv[]) {

        pthread_t tensarThreadId;
        pthread_create(&tensarThreadId, NULL, tensarThreadFunc, 0);

        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGB);
        glutInitWindowSize(SCREEN_WIDTH, SCREEN_HEIGHT);
        glutInitWindowPosition(0, 0);
        glutCreateWindow("Tensar | MNIST dataset");
        glutDisplayFunc(display);
        glutReshapeFunc(reshape);
        glutMouseFunc(mouse);
        glutSpecialFunc(keyboard);
        glutIdleFunc(idle);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        //glEnable(GL_LINE_SMOOTH);

        glutMainLoop();

        return 0;
}
