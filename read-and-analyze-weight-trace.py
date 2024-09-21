import os
import numpy as np

def read_all_weights_from_file(file):
    # Load the file
    data = np.load(file, allow_pickle=True)
    # Loop through all the stored arrays
    all_data = []
    total_length = 0
    for key in data.files:
        the_data = data[key]
        flattened_data = the_data.flatten()
        total_length += len(flattened_data)
        all_data.extend(flattened_data)
    data.close()
    return all_data

def consolidate_weights_from_directory(directory, output_file):
    files = [f for f in os.listdir(directory) if f.endswith('.npz')]
    first_file = f"{directory}/{files[0]}"
    first_row = read_all_weights_from_file(first_file)
    if first_row is None:
        raise ValueError('No weights found in the first file')
    columns = len(first_row)
    rows = len(files)
    consolidated_weights = np.zeros((rows, columns))
    for i, file in enumerate(files):
        consolidated_weights[i] = read_all_weights_from_file(f"{directory}/{file}")
    np.savez(output_file, consolidated_weights)



if __name__ == '__main__':
    consolidate_weights_from_directory('data', 'consolidated_weights.npz')