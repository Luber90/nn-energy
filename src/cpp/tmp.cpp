#include <iostream>
#include <filesystem>
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
//#include <torchvision/vision.h>
#include <random>

namespace fs = std::filesystem;




int main(int argc, char** argv){
    
    
    for(const auto& batch : *train_loader){
        std::cout << batch.data << std::endl;
        break;
    }
    // for(auto i : paths){
    //     std::cout << i << '\n';
    // }
    
    // double a = 1024.5;
    // uint64_t* b = static_cast<uint64_t*>(static_cast<void*>(&a));
    // std::cout << "0x" << std::hex << *b << std::endl;

    // if (argc != 2) {
    //     printf("usage: DisplayImage.out <Image_Path>\n");
    //     return -1;
    // }
    // Mat image;
    // image = imread(argv[1], 1);
    // if (!image.data) {
    //     printf("No image data \n");
    //     return -1;
    // }
    // namedWindow("Display Image", WINDOW_AUTOSIZE);
    // imshow("Display Image", image);
    // waitKey(0);
    return 0;
}