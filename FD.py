import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from tqdm import tqdm

def finiteDifferences(f_arr, x_arr):
    '''
    Function to implement a finite differences differentiation scheme for numerical data. 
 
    '''
    f_deriv = np.zeros_like(f_arr)

    f_deriv[0] = (f_arr[1] - f_arr[0]) / (x_arr[1] - x_arr[0])  # special case: first element -> forward difference
    f_deriv[-1] = (f_arr[-1] - f_arr[-2]) / (x_arr[-1] - x_arr[-2]) # special case: last element -> backwards difference
    f_deriv [1:-1] = (f_arr[2:] - f_arr[:-2]) / (x_arr[2:] - x_arr[:-2]) # bulk of data -> central difference
    
    return f_deriv
