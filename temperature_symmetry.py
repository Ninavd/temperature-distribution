# does the same as temperature.py but uses the symmetry in x of the system

import copy
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

# square with 9x9 inches, lower 4 inches under water, 
# mid section 3x3 inches packed with rye

def update_row(new, current, i, start_col, end_col):
    """
    Updates new[i][start_col:end_col] in new using the current distribution.
    Can not do negative indeces
    """
    if end_col == new.shape[1]:
        end_col = end_col - 1
        new[i][end_col] = (
                current[i - 1][end_col]       # value above
                + current[i + 1][end_col]     # value below
                + current[i][end_col - 1]     # value to left
                ) / 3

    new[i][start_col:end_col] = (
            current[i - 1][start_col:end_col]       # row above
            + current[i + 1][start_col:end_col]     # row below
            + current[i][start_col + 1:end_col + 1] # row shifted to right
            + current[i][start_col - 1:end_col - 1] # row shifted to left
            ) / 4


def init_temperature(DIMENSION, water_temp, rye_temp, top_temp):
    """Initializes initial state of temperature distribution."""

    rye_start = int(DIMENSION / 3) # same for col and row
    rye_end_row = 2 * rye_start + 1
    rye_end_col = DIMENSION - 1

    idx_waterline = int(DIMENSION / 9 * 4)

    # boundary values 
    temperatures[-1, :] = water_temp
    temperatures[0, :] = top_temp
    temperatures[rye_start:rye_end_row, rye_start:rye_end_col] = rye_temp

    # constant part of sides
    temperatures[idx_waterline:, 0] = water_temp

    # linear part of sides
    delta_temp = (top_temp - water_temp) / (idx_waterline) 
    linear_temps = np.ones(idx_waterline) * delta_temp
    linear_temps = np.cumsum(linear_temps) + water_temp
    temperatures[:idx_waterline, 0] = linear_temps[::-1]

    # initialize rest as value between 32 and 212
    temperatures[temperatures == 0] = 90

    return temperatures, rye_start, rye_end_col, rye_end_row

def save_to_csv(df, current, path):
    """Saves the coordinates & flattened array to csv"""
    df['T'] = current.flatten()
    df.to_csv(path, index=False)

if __name__=="__main__":
    
    DIMENSION = 256
    n_cols = DIMENSION // 2
    temperatures = np.zeros((DIMENSION, n_cols))

    water_temp = 32
    rye_temp = 212
    top_temp = 100 

    temperatures, rye_start, rye_end_col, rye_end_row = init_temperature(DIMENSION, water_temp, rye_temp, top_temp)

    boundary_mask = temperatures != 90
    boundary_values = temperatures[boundary_mask]

    plt.imshow(temperatures)
    plt.title('Initial state')
    plt.colorbar()
    plt.show()

    # create flattened vectors with all coordinate points
    rows, cols = temperatures.shape
    x_coords, y_coords = np.meshgrid(range(cols), range(rows))
    x_coords, y_coords = x_coords.flatten(), np.flipud(y_coords).flatten()

    # create template dataframe for saving to csv
    df = pd.DataFrame({
            'xs':x_coords, 
            'ys':y_coords, 
            'T' :temperatures.flatten()
            })

    # initial values
    iters = 0
    frame = 0
    current = temperatures

    # laplace implementation
    tolerance = 10 ** -4
    max_diff = 100

    while max_diff > tolerance:

        # save every 50th frame to csv
        if iters % 50 == 0:
            path = f'symmetric_frames/frame_{frame}.csv'
            save_to_csv(df, current, path)
            frame += 1

        new = copy.copy(current)

        # dont update the boundaries!! NOTE: most right column is not a constant boundary anymore
        for i in range(1, rows - 1):

            # TODO: check if for loop faster instead of slicing?
            if i < rye_start or i >= rye_end_row:
                update_row(new, current, i, start_col=1, end_col=cols)
            else:
                update_row(new, current, i, start_col=1, end_col=rye_start)

        # for stopping condition
        max_diff = np.max(np.absolute(new - current))

        current = new

        iters += 1

    # save final frame
    save_to_csv(df, current, f'symmetric_frames/frame_{frame}.csv')

    print(f'completed iterations before stopping {iters}')

    # for testing
    plt.imshow(current, cmap='inferno')
    plt.colorbar()
    plt.show()