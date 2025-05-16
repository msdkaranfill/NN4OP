/**
 * @file main.cpp
 * @author Krystof Teissing (k.teissing@gmail.com)
 * @version 0.1
 * @date 2023-05-25
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <Eigen/Eigen>
#include <chrono>
#include <iomanip>
#include <memory>
#include <fcntl.h>

#include "pmm_trajectory3d.hpp"
#include "yaml-cpp/yaml.h"
#include "pmm_mg_trajectory3d.hpp"
#include "common.hpp"
#include "cftg.hpp"
#include "NN_model.hpp"
#include <iomanip>

using namespace pmm;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void signal_callback(int sig) {
  std::cout << "Signal " << sig << " received" << std::endl;
  exit(sig);
}

//std::pair<Vector<3>, Scalar> generate_trajectory(Vector<3> p_s, Vector<3> v_s, Vector<3> p_1, Vector<3> p_e, Vector<3> v_e, bool debug=false) {
//multiple gates:
std::pair<Vector<3>, Scalar> generate_trajectory(Vector<3> p_s, Vector<3> v_s, 
Vector<3> p_1, Vector<3> p_e, Vector<3> v_e, bool debug=false) {

  // Velocity Optimization parameters --------------------------------------------------------------/
  // Thrust decomposition
  const int TD_max_iter = 10;
  const Scalar TD_acc_precision = 1e-3;
  const Scalar max_acc_norm = 1.0;
  const Scalar max_velocity_per_axis = 4.0 / sqrt(3);

  Vector<3> max_per_axis_acc_vec;
  Vector<3> min_per_axis_acc_vec;
  std::tie(min_per_axis_acc_vec, max_per_axis_acc_vec) = compute_per_axis_acc_limits_from_limited_thrust(max_acc_norm);

  // Gradient Method
  Scalar alpha = 33.0;
  Scalar alpha_reduction_factor = 0.45;
  Scalar alpha_min_threshold = 1e-3;
  double dT_precision = 1e-3;
  int max_iter = 30;
  int single_axis_optim_idx = -1; // switch between single axis and all axis optimization

  if (debug)
    std::cout << "--------------------------- SETTING UP WAYPOINTS ------------------------" << std::endl;
  std::vector<Vector<3>> path_waypoints;
  path_waypoints.push_back(p_s);
  path_waypoints.push_back(p_1);
  path_waypoints.push_back(p_e);

  if (debug)
    std::cout << "-------------------------- COMPUTING TRAJECTORY --------------------------" << std::endl;

  // Initialize velocities in waypoints -------------------/
  std::vector<Vector<3>> gates_velocities = compute_initial_velocities(path_waypoints, v_s, v_e, max_per_axis_acc_vec[0], max_velocity_per_axis);

  // Create trajectory object
  PMM_MG_Trajectory3D mp_tr(path_waypoints, gates_velocities, min_per_axis_acc_vec, max_per_axis_acc_vec, max_velocity_per_axis);

  if (debug) {
    mp_tr.exportTrajectoryToCSV("scripts/trajectory_data/dp_example_mp_tr_orig.csv");
    mp_tr.sample_and_export_trajectory(1e-3,"scripts/trajectory_data/dp_example_mp_tr_orig_sampl.csv");
    std::cout << "Original trajectory duration: " << mp_tr.duration() << " s." << std::endl;
  }

  // Optimize velocities
  mp_tr.optimize_velocities_at_positions(alpha, alpha_reduction_factor, alpha_min_threshold,
                                        max_iter, dT_precision, single_axis_optim_idx, false,
                                        false, "scripts/trajectory_data/dp_example_mp_tr_data.csv", false);

  if (debug)                                      
    std::cout << "Optimized trajectory duration after first optimization run: " << mp_tr.duration()  << " s." << std::endl;

  // Recompute using Thrust Decomposition
  std::vector<Vector<3>> v = mp_tr.get_current_velocities();
  PMM_MG_Trajectory3D mp_tr_o(path_waypoints, v, max_acc_norm, max_velocity_per_axis, TD_max_iter, TD_acc_precision, false);
  if (debug) {
    mp_tr_o.exportTrajectoryToCSV("scripts/trajectory_data/dp_example_mp_tr_optim.csv");
    mp_tr_o.sample_and_export_trajectory(1e-3,"scripts/trajectory_data/dp_example_mp_tr_optim_sampl.csv");
    std::cout << "Optimized trajectory duration after TD: " << mp_tr_o.duration()  << " s." << std::endl;
  }

  // Second optimization run
  mp_tr_o.optimize_velocities_at_positions(alpha, alpha_reduction_factor, alpha_min_threshold,
                                          max_iter, dT_precision, single_axis_optim_idx, false,
                                          false, "scripts/trajectory_data/dp_example_mp_tr_data_2nd_run.csv", false);
  if (debug) {
    mp_tr_o.exportTrajectoryToCSV("scripts/trajectory_data/dp_example_mp_tr_optim_2nd_run.csv");
    mp_tr_o.sample_and_export_trajectory(1e-3,"scripts/trajectory_data/dp_example_mp_tr_optim_sampl_2nd_run.csv");
    std::cout << "Optimized trajectory duration after second optimization run with TD: " << mp_tr_o.duration()  << " s." << std::endl;

    std::cout << "Visualize results by running ./scripts/plot_results.py!"<< std::endl;
    std::cout << "#----------------------------------------- END ------------------------------------------#" << std::endl << std::endl;
  }
  

  std::vector<Vector<3>> velocities = mp_tr_o.get_current_velocities();
  Vector<3> velocity = velocities[1];
  Scalar duration = mp_tr_o.duration();

  std::pair<Vector<3>, Scalar> result = {velocity, duration};

  return result;
}

int main(int argc, char** argv) {
  // Load config file
  std::string config_file = "config.yaml";
  YAML::Node config = YAML::LoadFile(config_file);
  
  // Get inputs from config
  Vector<3> p_s;
  Vector<3> v_s;
  config["start"]["position"] >> p_s;
  config["start"]["velocity"] >> v_s;
  // goal position and velocity
  Vector<3> p_e;
  Vector<3> v_e;
  config["end"]["position"] >> p_e;
  config["end"]["velocity"] >> v_e;
  // waypoint position
  Vector<3> p_1;    
  config["gates"] >> p_1;
  /*
  std::cout << "Input parameters:" << std::endl;
  std::cout << "Start position: " << p_s.transpose() << std::endl;
  std::cout << "Start velocity: " << v_s.transpose() << std::endl;
  std::cout << "Gate position: " << p_1.transpose() << std::endl;
  std::cout << "End position: " << p_e.transpose() << std::endl;
  std::cout << "End velocity: " << v_e.transpose() << std::endl;*/






  // Path to the neural network model
  const std::string nn_path = "traced_model.pt";
    // Load the model
  auto& model = load_model(nn_path);
  // Prepare inputs for the neural network
  std::vector<double> nn_inputs = {
    p_s.x(), p_s.y(), p_s.z(),    // Start position
    v_s.x(), v_s.y(), v_s.z(),    // Start velocity
    p_e.x(), p_e.y(), p_e.z(),    // End position
    v_e.x(), v_e.y(), v_e.z(),    // End velocity
    p_1.x(), p_1.y(), p_1.z()     // Gate position
  };

  // Start the neural network prediction process
  //std::cout << "Starting neural network prediction using direct C++ model loading..." << std::endl;
  auto model_process_start = std::chrono::high_resolution_clock::now();
  
  // Run the optimization process
  //std::cout << "Running optimization..." << std::endl;
  auto opt_start = std::chrono::high_resolution_clock::now();
  std::pair<Vector<3>, Scalar> opt_result = generate_trajectory(p_s, v_s, p_1, p_e, v_e, false);
  auto opt_end = std::chrono::high_resolution_clock::now();
  auto opt_duration = std::chrono::duration_cast<std::chrono::milliseconds>(opt_end - opt_start);

  // Make prediction
  auto model_result = predict_with_NN(model, nn_inputs);
  auto model_end = std::chrono::high_resolution_clock::now();
  auto model_duration = std::chrono::duration_cast<std::chrono::milliseconds>(model_end - model_process_start);

  // Output in the format expected by visualize_trajectory.py
  // Format: start_pos(3) start_vel(3) end_pos(3) end_vel(3) gate_pos(3) vel_opt(3) time_opt vel_nn(3) time_nn 
  std::cout << std::fixed << std::setprecision(2)
            << p_s.transpose() << " " << v_s.transpose() << " " << p_e.transpose() << " " << v_e.transpose() << " " << p_1.transpose() << " " << opt_result.first.transpose() << " " << opt_result.second << " " << model_result.first.transpose() << " " << model_result.second << std::endl;
           
}
