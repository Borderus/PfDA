# Import libraries

from pandas import Series, DataFrame

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import re

# *************************************************************

# Define functions to print Series and Frames


def printSeries(num1, num2):
    exec(f'print("Series{num1}.{num2}:", "", series{num1}_{num2}, sep="""\n""")')
    print()

def printFrame(num1, num2):
    exec(f'print("DataFrame{num1}.{num2}:", "", frame{num1}_{num2}, sep="""\n""")')
    print()

# *************************************************************

# ******************* Modelling Libraries *********************
    

# First trick is that most modelling libraries use numpy arrays -
# we can convert a dataframe using .to_numpy()
    
frame1_1 = pd.DataFrame({
    'x0': [1, 2, 3, 4, 5],
    'x1' : [0.01, -0.01, 0.25, -4.1, 0.],
    'y': [-1.5, 0, 3.6, 1.3, -2]
})

printFrame(1,1)

array1_2 = frame1_1.to_numpy()
print('ND Array 1.2:\n', array1_2, '\n')

# And back:
frame1_3 = pd.DataFrame(array1_2, columns=['x0', 'x1', 'y'])
printFrame(1,3)