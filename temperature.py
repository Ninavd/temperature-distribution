import copy
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

def update_row(new, current, i, start_col, end_col):
    """
    Updates new[i][start_col:end_col] in new using the current distribution.
    Can not do negative indeces
    """
    new[i][start_col:end_col] = (
            current[i - 1][start_col:end_col]       # row above
            + current[i + 1][start_col:end_col]     # row below
            + current[i][start_col + 1:end_col + 1] # row shifted to right
            + current[i][start_col - 1:end_col - 1] # row shifted to left
            ) / 4

def save_to_csv(df, current, path):
    """Saves the coordinates & flattened array to csv"""
    df['T'] = current.flatten()
    df.to_csv(path, index=False)

if __name__=="__main__":

    # square with 9x9 inches, lower 4 inches under water, 
    # mid section 3x3 inches packed with rye

    DIMENSION = 256
    temperatures = np.zeros((DIMENSION, DIMENSION))

    water_temp = 32
    rye_temp = 212
    top_temp = 100 

    rye_start = int(DIMENSION / 3)
    rye_end = 2 * rye_start + 1
    idx_waterline = int(DIMENSION / 9 * 4)

    # boundary values 
    temperatures[-1, :] = water_temp
    temperatures[0, :] = top_temp
    temperatures[rye_start:rye_end, rye_start:rye_end] = rye_temp

    # constant part of sides
    temperatures[idx_waterline:, 0] = water_temp
    temperatures[idx_waterline:, -1] = water_temp

    # linear part of sides
    delta_temp = (top_temp - water_temp) / (idx_waterline) 
    linear_temps = np.ones(idx_waterline) * delta_temp
    linear_temps = np.cumsum(linear_temps) + water_temp
    temperatures[:idx_waterline, 0] = linear_temps[::-1]
    temperatures[:idx_waterline, -1] = linear_temps[::-1]

    boundary_mask = temperatures > 0
    boundary_values = temperatures[boundary_mask]

    # initialize rest as value between 32 and 212
    temperatures[temperatures == 0] = 90 

    plt.imshow(temperatures)
    plt.title('Initial state')
    plt.colorbar()
    plt.show()

    # laplace implementation
    tolerance = 10 ** -4
    max_diff = 100
    current = temperatures

    # create flattened vectors with all coordinate points
    x_coords, y_coords = np.meshgrid(range(DIMENSION), range(DIMENSION))
    x_coords, y_coords = x_coords.flatten(), y_coords.flatten()

    # TODO: add padding to make boundary values clearer (at the end! otherwise copying is expensive)
    # create template dataframe for saving to csv
    df = pd.DataFrame({
            'xs':x_coords, 
            'ys':y_coords, 
            'T' :temperatures.flatten()
            })

    iters = 0
    frame = 0
    while max_diff > tolerance:

        # save every 50th frame to csv
        # if iters % 50 == 0:
        #     path = f'frames/frame_{frame}.csv'
        #     save_to_csv(df, current, path)
        #     frame += 1

        new = copy.copy(current)

        # dont update the boundaries!!
        for i in range(1, DIMENSION - 1):
            # TODO: check if for loop faster instead of slicing?
            if i < rye_start or i >= rye_end:
                update_row(new, current, i, start_col=1, end_col=DIMENSION - 1)
            else:
                update_row(new, current, i, start_col=1, end_col=rye_start)
                update_row(new, current, i, start_col=rye_end, end_col=DIMENSION - 1)

        # maximum difference between new and old values
        max_diff = np.max(np.absolute(new - current))

        current = new

        iters += 1

    # save final frame
    # path = f'frames/frame_{frame}.csv'
    # save_to_csv(df, current, path)

    print(f'completed iterations before stopping {iters}')

    plt.imshow(current, cmap='inferno')
    plt.colorbar()
    plt.show()