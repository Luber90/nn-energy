#include <iostream>
#include <filesystem>
#include <torch/torch.h>
#include "unet.h"
#include "utils.h"
#include <memory>
#include <filesystem>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>

namespace fs = std::filesystem;

template <typename T>
void train_one_epoch(UNet model, torch::optim::RMSprop& optimizer,
                     T const &train_loader, size_t epoch){
    model->train();
    int batch_index = 0;
    double running_loss = 0.0;
    for(const auto& batch : *train_loader){
        optimizer.zero_grad();
        torch::Tensor prediction = model->forward(batch.data.to(torch::TensorOptions().device(torch::kCUDA).pinned_memory(true)));
        torch::Tensor loss = torch::mse_loss(prediction, batch.target.to(torch::TensorOptions().device(torch::kCUDA).pinned_memory(true)));
        loss.backward();
        optimizer.step();
        running_loss += loss.item<double>();
        if (batch_index++ % 200 == 199) {
            std::cout << "[" << epoch+1 << ",   " << batch_index << "]" << " loss: " << running_loss / 200. << std::endl;
            running_loss = 0.0;
        }
    }
}

template<typename T>
void check_validation(UNet model, torch::optim::RMSprop& optimizer,
                      T const &val_loader, size_t epoch, const int dataset_size, const int split){
    
    torch::NoGradGuard no_grad;
    model->eval();
    double val_loss = 0.0;

    for(const auto& val_batch : *val_loader){
        torch::Tensor prediction = model->forward(val_batch.data.to(torch::TensorOptions().device(torch::kCUDA).pinned_memory(true)));
        torch::Tensor loss = torch::mse_loss(prediction, val_batch.target.to(torch::TensorOptions().device(torch::kCUDA).pinned_memory(true)));
        val_loss += loss.item<double>();
    }
    val_loss /= (static_cast<double>(dataset_size-split)/128.);
    std::cout << "[" << epoch+1 << "]" << " validation loss: " << val_loss << std::endl;
}

int main(){
    auto photos_iter = fs::directory_iterator("/workspace/unlabeled2017");
    std::vector<fs::path> paths = {};
    const int dataset_size =  60000, split = static_cast<int>(dataset_size*0.8);
    const int epochs = 5;

    for(auto i : photos_iter){
        paths.push_back(i);
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(paths.begin(), paths.end(), g);
    
    std::vector<fs::path> train_paths = std::vector(paths.begin(), paths.begin() + split);
    std::vector<fs::path> test_paths = std::vector(paths.begin() + split, paths.begin() + dataset_size);

    std::cout << dataset_size << std::endl;
    std::cout << train_paths.size() << " " << test_paths.size() << std::endl;

    const std::uint8_t batch_size = 32;

    auto train_set = ColorizationDataset(std::move(train_paths)).map(torch::data::transforms::Stack<>());
    auto train_loader = 
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), torch::data::DataLoaderOptions().batch_size(batch_size).workers(16)
            );

    auto val_set = ColorizationDataset(std::move(test_paths)).map(torch::data::transforms::Stack<>());
    auto val_loader = 
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_set), torch::data::DataLoaderOptions().batch_size(128).workers(16)
        );

    auto start_time = std::chrono::system_clock::now();

    UNet model;
    model->to(torch::kCUDA);

    torch::optim::RMSprop optimizer(model->parameters(), /*lr=*/0.0001);

    for(size_t epoch = 0; epoch < epochs; epoch++){
        train_one_epoch(model, optimizer, train_loader, epoch);

        check_validation(model, optimizer, val_loader, epoch, dataset_size, split);  
    }
    auto end_time = std::chrono::system_clock::now();
    auto start = std::chrono::system_clock::to_time_t(start_time);
    auto end = std::chrono::system_clock::to_time_t(end_time);
    std::cout << ctime(&start) << std::endl;
    std::cout << ctime(&end) << std::endl;
    
    return 0;
}