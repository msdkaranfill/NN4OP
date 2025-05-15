#pragma once

#include <torch/script.h>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>

// Graph data structure to match PyTorch Geometric's Data class
struct GraphData {
    torch::Tensor x;  // Node features
    torch::Tensor edge_index;  // Edge indices
    torch::Tensor edge_attr;  // Edge features
    torch::Tensor gate_indices;  // Gate node indices
    int num_nodes;
};

class GraphDataTransformer {
public:
    GraphDataTransformer();
    GraphData transform(const std::vector<double>& inputs);
};

// Function to load the model once and reuse it
torch::jit::script::Module& load_model(const std::string& nn_path);

// Function to predict with the model
std::pair<Eigen::Vector3d, double> predict_with_NN(
    torch::jit::script::Module& model, 
    const std::vector<double>& inputs
); 