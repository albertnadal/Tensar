#!/bin/bash
rm -rf out/
mkdir out
echo "Compiling tensor.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/tensor.cpp.o -c src/tensor.cpp
echo "Compiling tensor_float.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/tensor_float.cpp.o -c src/tensor_float.cpp
echo "Compiling tensor_gradient.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/tensor_gradient.cpp.o -c src/tensor_gradient.cpp
echo "Compiling gradient.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/gradient.cpp.o -c src/gradient.cpp
echo "Compiling layer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/layer.cpp.o -c src/layer.cpp
echo "Compiling convolutional_layer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/convolutional_layer.cpp.o -c src/convolutional_layer.cpp
echo "Compiling relu_layer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/relu_layer.cpp.o -c src/relu_layer.cpp
echo "Compiling pool_layer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/pool_layer.cpp.o -c src/pool_layer.cpp
echo "Compiling fully_connected_layer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/fully_connected_layer.cpp.o -c src/fully_connected_layer.cpp
echo "Compiling input_case.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/input_case.cpp.o -c src/input_case.cpp

echo "Compiling tensor_render_frame_buffer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/tensor_render_frame_buffer.cpp.o -c src/tensor_render_frame_buffer.cpp
echo "Compiling layer_grid_frame_buffer.cpp"
g++ -std=c++11 -stdlib=libc++ -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wno-deprecated -o out/layer_grid_frame_buffer.cpp.o -c src/layer_grid_frame_buffer.cpp
echo "Compiling NeuralNetwork"
g++ -std=c++11 -stdlib=libc++ out/input_case.cpp.o out/fully_connected_layer.cpp.o out/pool_layer.cpp.o out/tensor_gradient.cpp.o out/tensor_float.cpp.o out/tensor.cpp.o out/gradient.cpp.o out/tensor_render_frame_buffer.cpp.o NeuralNetworkMNIST.cpp out/layer_grid_frame_buffer.cpp.o -o NeuralNetwork -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.13.sdk -Wl,-search_paths_first -Wl,-headerpad_max_install_names -framework OpenGL -framework OpenGL -framework GLUT -framework Cocoa -Wno-deprecated
