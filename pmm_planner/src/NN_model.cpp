#include <iostream>
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "NN_model.hpp"

// Function to load the model once and reuse it
torch::jit::script::Module& load_model(const std::string& nn_path) {
    static torch::jit::script::Module model;
    static std::once_flag load_flag;
    std::call_once(load_flag, [&]() {
        try {
            //std::cout << "Loading model from: " << nn_path << std::endl;
            model = torch::jit::load(nn_path);
            model.eval();  // Set to evaluation mode
            //std::cout << "Model loaded successfully" << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model: " << e.what() << std::endl;
            std::exit(EXIT_FAILURE);
        }
    });
    return model;
}

// Prepare input tensors for the model
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> prepare_model_input(
    const std::vector<double>& inputs
) {
    if (inputs.size() != 15) {
        throw std::invalid_argument("Expected 15 input features, got " + std::to_string(inputs.size()));
    }

    // Convert inputs to tensor
    torch::Tensor input_tensor = torch::from_blob(
        const_cast<double*>(inputs.data()),
        {1, static_cast<long>(inputs.size())},
        torch::kDouble
    ).clone();

    // Extract features
    auto start_pos = input_tensor.slice(1, 0, 3);    // [1, 3]
    auto start_vel = input_tensor.slice(1, 3, 6);    // [1, 3]
    auto end_pos = input_tensor.slice(1, 6, 9);      // [1, 3]
    auto end_vel = input_tensor.slice(1, 9, 12);     // [1, 3]
    auto gate_pos = input_tensor.slice(1, 12, 15);   // [1, 3]

    // Create node features: [3 nodes, 6 features each]
    // Flatten each [1, 3] tensor to [3] before concatenation
    auto start_node = torch::cat({start_pos.squeeze(0), start_vel.squeeze(0)});       // [6]
    auto gate_node  = torch::cat({gate_pos.squeeze(0), torch::zeros(3, torch::kDouble)});  // [6]
    auto end_node   = torch::cat({end_pos.squeeze(0), end_vel.squeeze(0)});          // [6]

    // Stack nodes into [3, 6] tensor
    auto x = torch::stack({start_node, gate_node, end_node});  // [3, 6]

    // Convert to float32 as the model expects float32 input
    x = x.to(torch::kFloat32);

    // Create edge indices
    auto edge_index = torch::tensor({{0, 1}, {1, 2}}, torch::kLong).t();  // [2, 2]

    // Create edge features
    auto start_to_gate = torch::cat({gate_pos - start_pos, -start_vel}, 1).squeeze(0);  // Start -> Gate
    auto gate_to_end = torch::cat({end_pos - gate_pos, end_vel}, 1).squeeze(0);  // Gate -> End
    auto edge_attr = torch::stack({start_to_gate, gate_to_end}, 0);  // [2, 6]
    
    // Convert to float32 as the model expects float32 input
    edge_attr = edge_attr.to(torch::kFloat32);

    // Create gate indices
    auto gate_indices = torch::tensor({1}, torch::kLong);  // Gate is at index 1

#ifdef DEBUG
    std::cout << "Node features:\n" << x << std::endl;
    std::cout << "Edge index:\n" << edge_index << std::endl;
    std::cout << "Edge attributes:\n" << edge_attr << std::endl;
    std::cout << "Gate indices:\n" << gate_indices << std::endl;
#endif

    return std::make_tuple(x, edge_index, edge_attr, gate_indices);
}

// Function to predict with the model - simplified to assume CPU execution and fixed output format
std::pair<Eigen::Vector3d, double> predict_with_NN(
    torch::jit::script::Module& model, 
    const std::vector<double>& inputs
) {
    try {
        // Prepare input tensors for the model
        auto [x, edge_index, edge_attr, gate_indices] = prepare_model_input(inputs);

        // Reshape edge_index to match the model's expectation
        edge_index = edge_index.unsqueeze(1);  // Add batch dimension [2, 1, 2]

        // Prepare input for the model
        std::vector<torch::jit::IValue> model_inputs {
            x,
            edge_index,
            edge_attr,
            gate_indices
        };

        // Run inference with no gradients
        torch::NoGradGuard no_grad;
            
        // Get model output (velocity and time)
        auto output = model.forward(model_inputs).toTuple();
        
        // Extract velocity (first element) and time (second element)
        auto velocity_tensor = output->elements()[0].toTensor();
        auto time_tensor = output->elements()[1].toTensor();
        
        // Assuming velocity is [batch_size=1, 3] and time is [batch_size=1, 1]
        // Create Eigen::Vector3d directly from the tensor data
        Eigen::Vector3d velocity(
            velocity_tensor[0][0].item<float>(),
            velocity_tensor[0][1].item<float>(),
            velocity_tensor[0][2].item<float>()
        );
        
        // Extract the scalar time value
        double time = time_tensor[0][0].item<float>();

        return std::make_pair(velocity, time);
    }
    catch (const c10::Error& e) {
        std::cerr << "Error running inference: " << e.what() << std::endl;
        throw;
    }
}

// Python integration functions - commented out but kept for reference

/*
// Function to start Python process for model execution
FILE* start_model_process(const std::string& model_path, const std::string& config_path) {
    // Command to run the Python model
    std::string command = "python3 model_server.py " + model_path + " " + config_path;
    
    // Execute the command and return the pipe for later reading
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Could not open pipe to Python process");
    }
    
    return pipe;
}

// Function to get results from the Python model process
std::pair<Eigen::Vector3d, double> get_model_results(FILE* pipe) {
    // Read output from the command
    char buffer[1024];
    std::string output = "";
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    
    // Close pipe and check for errors
    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Python model execution failed with status: " + std::to_string(status));
    }
    
    // Parse JSON output
    Eigen::Vector3d velocity;
    double time = 0.0;
    
    // Very simple JSON parsing
    size_t vel_start = output.find("[");
    size_t vel_end = output.find("]");
    if (vel_start != std::string::npos && vel_end != std::string::npos) {
        std::string vel_str = output.substr(vel_start + 1, vel_end - vel_start - 1);
        std::stringstream ss(vel_str);
        std::string token;
        int i = 0;
        while (std::getline(ss, token, ',') && i < 3) {
            velocity(i) = std::stod(token);
            i++;
        }
    }
    
    size_t time_start = output.find("\"time\":");
    if (time_start != std::string::npos) {
        size_t value_start = output.find_first_of("0123456789.-", time_start);
        size_t value_end = output.find_first_not_of("0123456789.-", value_start);
        std::string time_str = output.substr(value_start, value_end - value_start);
        time = std::stod(time_str);
    }
    
    return {velocity, time};
}
*/

  
