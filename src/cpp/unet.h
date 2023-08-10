#pragma once

#include <torch/torch.h>

using namespace torch;

class A {};

class B : A {};

class conv_blockImpl : public nn::Module {
public:
    conv_blockImpl() : conv_blockImpl(1, 1) {}
    conv_blockImpl(int64_t in_c, int64_t out_c) :
        conv1(nn::Conv2dOptions(in_c, out_c, 3).stride(1).padding(1)),
        conv2(nn::Conv2dOptions(out_c, out_c, 3).stride(1).padding(1)),
        bn1(out_c),
        bn2(out_c)
        {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("bn1", bn1);
            register_module("bn2", bn2);
        }
    Tensor forward(const Tensor& input);
    Tensor operator()(const Tensor& input) {return this->forward(input);}
    nn::Conv2d conv1, conv2;
    nn::BatchNorm2d bn1, bn2;
};
TORCH_MODULE(conv_block);

class UNetImpl : public nn::Module {
public:
    UNetImpl() :
        e1(conv_block(1, 64)), e2(conv_block(64, 128)),
        e3(conv_block(128, 256)), e4(conv_block(256, 512)),
        mp(nn::MaxPool2d(nn::MaxPool2dOptions({2, 2}).stride({2, 2}))),
        b(conv_block(512, 1024)),
        ct1(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(1024, 512, 2).stride(2))),
        ct2(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(512, 256, 2).stride(2))),
        ct3(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(256, 128, 2).stride(2))),
        ct4(nn::ConvTranspose2d(nn::ConvTranspose2dOptions(128, 64, 2).stride(2))),
        d1(conv_block(1024, 512)), d2(conv_block(512, 256)), d3(conv_block(256, 128)),
        d4(conv_block(128, 64)), last(nn::Conv2d(nn::Conv2dOptions(64, 2, 3).stride(1).padding(1)))
    
    {
        register_module("e1", e1);
        register_module("e2", e2);
        register_module("e3", e3);
        register_module("e4", e4);
        register_module("b", b);
        register_module("d1", d1);
        register_module("d2", d2);
        register_module("d3", d3);
        register_module("d4", d4);
        register_module("ct1", ct1);
        register_module("ct2", ct2);
        register_module("ct3", ct3);
        register_module("ct4", ct4);
        register_module("last", last);
    }

    Tensor forward(const Tensor& input);

    conv_block e1, e2, e3, e4, b, d1, d2, d3, d4;
    nn::MaxPool2d mp;
    nn::ConvTranspose2d ct1, ct2, ct3, ct4;
    nn::Conv2d last;
};

TORCH_MODULE(UNet);