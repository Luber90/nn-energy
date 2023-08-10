#pragma once

#include <filesystem>
#include <vector>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

void show_image(cv::Mat& img, std::string title)
{
    cv::namedWindow(title, cv::WINDOW_NORMAL); // Create a window for display.
    cv::imshow(title, img);
    cv::waitKey(0);
}

class ColorizationDataset : public torch::data::datasets::Dataset<ColorizationDataset> {
    using Example = torch::data::Example<>;
    using Paths = std::vector<fs::path>;

    Paths paths;
    const int psize = 128;

public:
    ColorizationDataset(const Paths&& paths) : paths(paths){ }

    Example get(std::size_t index){
        std::string path = paths[index].string();
        auto mat = cv::imread(path);
        cv::resize(mat, mat, {psize, psize});
        cv::cvtColor(mat, mat, cv::COLOR_BGR2Lab);
        cv::Mat mat2;
        mat.convertTo(mat2, CV_32FC3);

        std::vector<cv::Mat> channels(3);
        cv::split(mat2, channels);

        auto L = torch::from_blob(
                                channels[0].data,
                                {psize, psize},
                                torch::kFloat);
        L = (L/128.)-1;
        auto A = torch::from_blob(
                                channels[1].data,
                                {psize, psize},
                                torch::kFloat);
        A = (A/128.)-1;
        auto B = torch::from_blob(
                                channels[2].data,
                                {psize, psize},
                                torch::kFloat);
        B = (B/128.)-1;
        L = L.view({1, psize, psize});
        auto AB = torch::cat({A, B}).view({2, psize, psize});

        return {L, AB};
    }

    torch::optional<std::size_t> size() const {
        return paths.size();
    }
};