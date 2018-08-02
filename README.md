## Tensar CNN

Tensar is an easy implementation written in C++11 to help you develop, understand and visualize simple Convolutional Neural Networks from scratch.

With the aim to view how tensors data evolves I decided to use OpenGL for fast 2D and 3D renderization of all the tensors in real time during the training process. The sample used in the current implementation is trained with the MNIST dataset for handwritten digit recognition. I will implement other datasets trainings and different model topologies in a future.

This application is intended to help you to better understand how [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network) work from a practical point of view. Based on the implementation [simple_cnn](https://github.com/can1357/simple_cnn) by [can1357](https://github.com/can1357).

Screenshot:

[![Tensar](http://www.lafruitera.com/tensar_screenshot.png)](http://www.lafruitera.com/tensar_screenshot.png)

Video:

[![Tensar](https://img.youtube.com/vi/mqp0HtEZdus/0.jpg)](https://www.youtube.com/watch?v=mqp0HtEZdus)

## Dependencies

- OpenGL/Glut is used to display all the tensors as fast as possible in real time avoiding the use of CPU resources during the network training.
- A C++11 compiler. I suggest g++ (Gnu C++ compiler) so this is the compiler used in the build script.

## Building

On linux or macosx compile the source coude by running the provided build script and then launch the application following the instructions described bellow.

```sh
$ cd Tensar
$ ./build.sh
$ ./NeuralNetwork
```

## Decoupling graphics and neural network code

Both graphics renderer and the neural network algorithms run on their own run loops. The application main loop is used for data visualization via OpenGL and a secondary run loop on a thread is used for the neural network.

The implementation of the neural network is decoupled from the data visualization (OpenGL graphics library) by using the middleware classes LayerGridFrameBuffer and TensorRenderFrameBuffer, shared memory, double buffers and mutex controllers to ensure all works properly.

No C++ macros are provided to completely disable the OpenGL code yet, so I hope I will add one in the next release. Meanwhile you can remove the graphic layer just by removing all the OpenGL code and build the application again.


## TODO

- Add a macro for a more complete and easy graphics decoupling.
- Accelerate code execution via GPU by using third party libraries like CUDA or OpenCL.
- Add more dataset samples for training.
- Add different neural network topologies.
- Improve human interaction and data visualization.

## License
 
The MIT License (MIT)

Copyright (c) 2018 Albert Nadal Garriga

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
