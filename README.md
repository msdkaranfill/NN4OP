# Neural Network for Trajectory Optimization of UAVs

A Python-powered Neural Network for Solutions to the Orienteering Problem for UAVs, focusing on trajectory optimization.

## Overview

This project implements a combined neural network model for predicting optimal velocity vectors and travel time for UAVs navigating through waypoints. The model uses a graph-based approach to process spatial relationships between start points, gates, and end points.

## Key Components

- **TrajectoryModel**: Combined model with both velocity vector and time prediction capabilities
- **GraphDataTransformer**: Transforms raw coordinate data into graph structure for processing
- **Custom Loss Functions**: Includes physics-aware weighted MSE loss and MAPE loss for accurate evaluation

## Requirements

The project requires Python 3.8+ and the following packages:
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

To install all dependencies:
```
pip install -r requirements.txt
```

## Usage

1. Prepare your trajectory data in the required format:
    Inputs: [start_position_x, _y, _z, start_velocity_x:0, _y:0, _z:0, end_position_x, _y, z, end_velocity_x:0, _y:0, _z:0, gate_position_x, _y, _z]
    Outputs: [velocity_vector_at_the_gate_x, _y, _z, time_to_reach_to_the_end(s)]
2. Train the model using:
```
python main.py
```

## Model Architecture

The model uses a combination of graph neural networks and transformer-like attention mechanisms to process spatial relationships between waypoints and predict optimal velocity vectors at the given gate location and time. 