#include "unet.h"

using namespace torch;

Tensor conv_blockImpl::forward(const Tensor& input){
    auto output = relu(bn1(conv1(input)));
    output = relu(bn2(conv2(output)));
    return output;
}

Tensor UNetImpl::forward(const Tensor& input){
    auto p1 = e1(input);
    auto p2 = e2(mp(p1));
    auto p3 = e3(mp(p2));
    auto p4 = e4(mp(p3));
    auto b = this->b(mp(p4));
    auto ct1 = this->ct1(b);
    auto d1 = this->d1(cat({ct1, p4}, 1));
    auto ct2 = this->ct2(d1);
    auto d2 = this->d2(cat({ct2, p3}, 1));
    auto ct3 = this->ct3(d2);
    auto d3 = this->d3(cat({ct3, p2}, 1));
    auto ct4 = this->ct4(d3);
    auto d4 = this->d4(cat({ct4, p1}, 1));
    auto outputs = last(d4);
    return tanh(outputs);

}