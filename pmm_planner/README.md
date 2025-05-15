# PMM Planner

A planning and trajectory optimization system for UAVs based on the Projected Multi-Modal (PMM) approach.

## Overview

The PMM planner generates smooth trajectories through waypoints while satisfying dynamic constraints of the vehicle. It utilizes the trained neural network from the main repository to produce optimal velocity vectors at gate locations.

## Structure

- `src/` - C++ source files for the planner
- `include/` - Header files
- `Makefile` - Build instructions
- `config.yaml` - Configuration file
- Python utilities:
  - `loaddata.py` - To collect outputs from cpp project in big sizes that will be true data for training and validation process later.
  - `loadoutputs.py` - Output processing to create Datasets with clear data. 
  - `visualize_trajectory.py` - Visualization tools

## Building

To build the project:

```bash
make
```

## Usage

After building, run the planner with:

```bash
./main
```

Configuration can be modified in `config.yaml`.

## Integration with Neural Network

This planner integrates with the neural network model from the main repository to generate optimized trajectories based on learned velocity vectors and total time. 