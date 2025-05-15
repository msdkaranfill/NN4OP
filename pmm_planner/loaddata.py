import re
import numpy as np

def read_and_filter_mixed_data(file_path, output_file_path):
    # List to store all the data entries
    all_data = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'start_p' in line:
                current_data = extract_data(line)
                line = next(file)
                if 'Output' in line:
                    opt_output = extract_data(line)
                line = next(file)
                if 'NN' in line:
                    nn_output = extract_data(line)
                    all_data.append(current_data + nn_output + opt_output)
    with open(output_file_path, 'w') as file:
        for entry in all_data:
            # Convert numerical values to strings and join them with a space
            entry_str = ' '.join(map(str, entry))
            file.write(entry_str + '\n')

    return all_data

def extract_data(line):
    pattern = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
    values = [float(match) for match in pattern.findall(line)]
    return values

def read_and_filter_data(file_path, output_file_path):
    # List to store all the data entries
    all_data = []

    with open(file_path, 'r') as file:
        for line in file:
            if 'start_p' in line:
                current_data = extract_data(line)
                line = next(file)
                if 'Output' in line:
                    opt_output = extract_data(line)
                    all_data.append(current_data + opt_output)

    with open(output_file_path, 'w') as file:
        for entry in all_data:
            # Convert numerical values to strings and join them with a space
            entry_str = ' '.join(map(str, entry))
            file.write(entry_str + '\n')

    return all_data



# Replace 'your_file.txt' with the actual path to your text file
file_path = 'output1742865810.974216750k.txt'
out = 'output52_50k.txt'
all_data = read_and_filter_data(file_path, out)
data_arr = np.array(all_data)
