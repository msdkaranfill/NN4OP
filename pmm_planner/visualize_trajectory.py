import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to read and parse data
def read_trajectory_data(filename):
    trajectories = []
    with open(filename, 'r') as file:
        for line in file:
            data = list(map(float, line.strip().split()))
            # Split the line data into respective components
            trajectory = {
                "start_position": np.array(data[:3]),
                "start_velocity": np.array(data[3:6]),
                "end_position": np.array(data[6:9]),
                "end_velocity": np.array(data[9:12]),
                "gate_position": np.array(data[12:15]),
                "velocity_opt": np.array(data[15:18]),
                "time_opt": data[18],
                "velocity_nn": np.array(data[19:22]),
                "time_nn": data[22]
            }
            trajectories.append(trajectory)
    return trajectories

# Function to plot trajectories
def plot_trajectories(trajectories):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for idx, traj in enumerate(trajectories[:10]):
        # Plot start and end positions
        ax.scatter(*traj["start_position"], color='blue', label='Start' if idx == 0 else "")
        ax.scatter(*traj["end_position"], color='orange', label='End' if idx == 0 else "")
        
        # Plot gate position
        ax.scatter(*traj["gate_position"], color='purple', label='Gate' if idx == 0 else "")
        
        # Plot velocity vectors from optimization process
        opt_arrow_start = traj["gate_position"]
        opt_arrow_end = opt_arrow_start + traj["velocity_opt"]
        ax.quiver(*opt_arrow_start, *traj["velocity_opt"], color='green', label='Opt Velocity' if idx == 0 else "")
        
        # Plot velocity vectors from neural network
        nn_arrow_start = traj["gate_position"]
        nn_arrow_end = nn_arrow_start + traj["velocity_nn"]
        ax.quiver(*nn_arrow_start, *traj["velocity_nn"], color='red', label='NN Velocity' if idx == 0 else "")
        
        # Annotate time outputs with formatted values
        opt_text = f"Opt: {traj['time_opt']:.2f}s, v=({traj['velocity_opt'][0]:.2f}, {traj['velocity_opt'][1]:.2f}, {traj['velocity_opt'][2]:.2f})"
        nn_text = f"NN: {traj['time_nn']:.2f}s, v=({traj['velocity_nn'][0]:.2f}, {traj['velocity_nn'][1]:.2f}, {traj['velocity_nn'][2]:.2f})"
        
        # Position the text slightly offset from the gate position
        text_offset = 0.2
        ax.text(traj["gate_position"][0], traj["gate_position"][1], traj["gate_position"][2] + text_offset, 
                opt_text, color='green', fontsize=4)
        ax.text(traj["gate_position"][0], traj["gate_position"][1], traj["gate_position"][2] - text_offset, 
                nn_text, color='red', fontsize=4)
    
    # Set plot details
    ax.set_title("Trajectory Visualization")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.legend(loc='best')
    
    # Save and show the plot
    plt.savefig("output_trajectory.png")
    plt.show()

# Main function to execute script
if __name__ == "__main__":
    filename = "outputs_comparison.txt"  # Input file name
    trajectories = read_trajectory_data(filename)
    plot_trajectories(trajectories)

