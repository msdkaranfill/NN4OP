import subprocess
import yaml
import random
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import random
import datetime

from tensorflow.python.ops.numpy_ops.np_array_ops import shape


def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
    return dumper.represent_list(array.tolist())

yaml.add_representer(np.ndarray, ndarray_representer)

# Function to generate a random point between minim and maxim
def generate_random_point(minim, maxim, p_s, p_e):
    if p_s is None:
        ret = []
        [ret.append(random.uniform(minim, maxim)) for _ in range(3)]
        return ret
    ret = []
    offset = random.randint(1,5)
    for i, j in zip(range(3),range(3)):
        minim = min(p_s[i], p_e[i])
        maxim = max(p_s[j], p_e[j])
        ret.append(random.uniform(max(minim+offset, -20), min(maxim-offset, 20)))
    return ret





listofconfigs = []
outputs = []
n_gates = 1
for i in range(10):
    # Load the YAML file
    with open(f'config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    config['start']['position'] = generate_random_point(-20, 20, None, p_e=None)
    p_s = config['start']['position']
    config['start']['velocity'] = [0, 0, 0]
    config['end']['position'] = generate_random_point(-20, 20, None, p_e=None)
    p_e = config['end']['position']
    config['end']['velocity'] = [0, 0, 0]
    #different gates for the same start and end params:
    gates = np.empty(shape=[n_gates, 3])
    gates = generate_random_point(-20, 20, p_s, p_e)
    config['gates'] = gates
    # Save the updated configuration back to the file
    with open(f'config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=None)


    _list = [p_s, config['start']['velocity'], p_e,
         config['end']['velocity'], config['gates']]
    #plot_one_set(_list)
    listofconfigs.append(_list)
    compile_result = subprocess.run('make')
    if compile_result.returncode == 0:
        print("Compilation successful. Running C++ program.")

    try:
        # Run the command and capture the output
        result = subprocess.run('./main', capture_output=True, universal_newlines=True)
        output = result.stdout
        # Print the C++ program's output
        #print("C++ Program Output:", output)
        outputs.append(output)
        print(f"Results have been appended to outputs for the {i}th time")

    except subprocess.CalledProcessError as e:
        print(f"Error running C++ program: {e}")


with open(f'outputs_comparison.txt', 'w') as file:
    [file.write(output) for output in outputs]
print(f"results have been save to the outputs_comparison.txt file")


def plotdata(datas):
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    i1 = random.randint(0, len(datas)-105)
    for data in datas[i1:i1+104]:
        #0-3:s_p, 6-9: e_p, 12-15:g_p
        start = data[:3]
        end = data[6:9]
        gate = data[12:15]
        ax.scatter(*start, c='r', marker='o', label='Start' if i == 0 else "")  # Start point in red
        ax.scatter(*end, c='g', marker='^', label='End' if i == 0 else "")  # End point in green
        ax.scatter(*gate, c='b', marker='x', label='Gate' if i == 0 else "")  # Gate point in blue
        plt.draw()
        plt.pause(0.5)

    # Display the legend once
    ax.legend()
    plt.show()


def plot_one_set(data):
    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111, projection='3d')
    # Labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #0-3:s_p, 6-9: e_p, 12-15:g_p
    data = np.array(data).reshape(-1,1)
    start = data[:3]
    end = data[6:9]
    gate = data[12:15]
    ax.scatter(*start, c='g', marker='o', label='Start' if i == 0 else "")  # Start point in green
    ax.scatter(*end, c='r', marker='^', label='End' if i == 0 else "")  # End point in red
    ax.scatter(*gate, c='b', marker='x', label='Gate' if i == 0 else "")  # Gate point in blue
    plt.draw()
    plt.pause(0.5)
    # Display the legend once
    ax.legend()
    plt.show()
