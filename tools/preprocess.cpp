#include "booster_vision/model/line_preprocessor.h"
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double Number(const std::string &value) {
    size_t end = 0;
    const double result = std::stod(value, &end);
    if (end != value.size() || !std::isfinite(result)) throw std::invalid_argument("Invalid number: " + value);
    return result;
}
int Integer(const std::string &value) {
    const double number = Number(value);
    if (number < 0 || number > 1000000 || std::floor(number) != number)
        throw std::invalid_argument("Expected integer in [0,1000000]: " + value);
    return static_cast<int>(number);
}
}

int main(int argc, char **argv) {
    try {
        std::map<std::string, std::string> options = {
            {"--input", ""}, {"--output", ""}, {"--variant", "original"},
            {"--clip-limit", "2.0"}, {"--tile-size", "8"}, {"--min-value", "140"},
            {"--max-saturation", "110"}, {"--strength", "0.9"},
            {"--repeat", "1"}, {"--warmup", "1"}};
        for (int i = 1; i < argc; ++i) {
            const std::string key = argv[i];
            if (key == "--help" || key == "-h") {
                std::cout << "hl_preprocess --input image --output image.png [--variant original|clahe|white|combined]\n"
                             "  [--clip-limit 2 --tile-size 8 --min-value 140 --max-saturation 110 --strength .9]\n"
                             "  [--warmup 1 --repeat 1]\n"
                             "stdout: JSON of Apply-only latency samples (ms) and effective parameters.\n";
                return 0;
            }
            if (!options.count(key) || i + 1 >= argc) throw std::invalid_argument("Unknown option or missing value: " + key);
            options[key] = argv[++i];
        }
        const auto variant = options.at("--variant");
        if (variant != "original" && variant != "clahe" && variant != "white" && variant != "combined")
            throw std::invalid_argument("Unknown variant: " + variant);
        booster_vision::LinePreprocessorConfig config;
        config.clahe_enabled = variant == "clahe" || variant == "combined";
        config.use_white_enhancement = variant == "white" || variant == "combined";
        config.clip_limit = Number(options.at("--clip-limit"));
        const int tile = Integer(options.at("--tile-size"));
        config.tile_grid_size = cv::Size(tile, tile);
        config.min_value = Integer(options.at("--min-value"));
        config.max_saturation = Integer(options.at("--max-saturation"));
        config.strength = Number(options.at("--strength"));
        const int repeat = Integer(options.at("--repeat"));
        const int warmup = Integer(options.at("--warmup"));
        if (repeat == 0) throw std::invalid_argument("repeat must be positive");
        if (options.at("--input").empty() || options.at("--output").empty())
            throw std::invalid_argument("--input and --output are required");
        booster_vision::LinePreprocessor preprocessor(config);
        const cv::Mat input = cv::imread(options.at("--input"), cv::IMREAD_COLOR);
        if (input.empty()) throw std::invalid_argument("Cannot decode input image");
        cv::Mat output;
        for (int i = 0; i < warmup; ++i) output = preprocessor.Apply(input);
        std::vector<double> samples;
        for (int i = 0; i < repeat; ++i) {
            const auto start = std::chrono::steady_clock::now();
            output = preprocessor.Apply(input);
            samples.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        }
        if (!cv::imwrite(options.at("--output"), output)) throw std::runtime_error("Cannot encode output image");
        std::cout << std::setprecision(12) << "{\"variant\":\"" << variant << "\",\"warmup\":" << warmup
                  << ",\"opencv\":\"" << CV_VERSION << "\",\"clip_limit\":" << config.clip_limit
                  << ",\"tile_size\":" << tile << ",\"min_value\":" << config.min_value
                  << ",\"max_saturation\":" << config.max_saturation << ",\"strength\":" << config.strength
                  << ",\"apply_ms\":[";
        for (size_t i = 0; i < samples.size(); ++i) std::cout << (i ? "," : "") << samples[i];
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "preprocess error: " << error.what() << '\n';
        return 2;
    }
}
