# Sample Data Directory

This directory should contain training data files for the trajectory optimization model.

## Expected Data Format

Each file should contain trajectories in the following format:
- Each line represents one trajectory sample
- Format: [start_position_x, _y, _z, start_velocity_x:0, _y:0, _z:0, end_position_x, _y, _z, end_velocity_x:0, _y:0, _z:0, gate_position_x, _y, _z, velocity_vector_at_the_gate_x, _y, _z, time_to_reach_to_the_end(s)]

## Sample File Naming Convention

Files are typically named with a pattern like `output<number>_<number_of_samples>k.txt`. For example:
- output10.txt
- output25_50k.txt (contains 50,000 trajectories)

## Data Generation

These files are not included in the repository due to their size. You can generate your own training data using appropriate simulations or obtain them from the authors. 