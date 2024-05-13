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

# ************* Aggregation and Group Operations **************
    

# ******************** GroupBy Mechanics **********************
    
frame1_1 = DataFrame({'key1': ['a', 'a', 'b', 'b', 'a'],
                'key2': ['one', 'two', 'one', 'two', 'one'],
                'data1': np.random.randn(5),
                'data2': np.random.randn(5)
                })
printFrame(1,1)

# Suppose we want to get the mean of the data1 column based one key1

# We can do this by using the GroupBy method to create a GroupBy object.
# This can then have its object method, mean() used
frame1_2 = frame1_1.groupby('key1').mean()
printFrame(1,2)

# We can group by numerous keys
frame1_3 = frame1_1.groupby(['key1', 'key2']).mean()
printFrame(1,3)

# We can get more creative off the back of this
frame1_4 = frame1_1.groupby(['key1', 'key2']).mean()['data1'].unstack()
printFrame(1,4)

# We can pass external arrays into the method:
years = [2002, 2003, 2004, 2002, 2003]
frame1_5 = frame1_1.groupby(years).mean()
printFrame(1,5)

# And we can count the groups with size():
frame1_6 = frame1_1.groupby('key2').size()
printFrame(1,6)

# ****************** Iterating Over Groups ********************

# The GroupBy object supports iteration

print('Iterating over keys:\n')

for name, group in frame1_1.groupby('key1'):
    print(name)
    print(group)
    print()

# And with compound keys:
    
print('Iterating over compound keys:\n')

for name, group in frame1_1.groupby(['key1', 'key2']):
    print(name)
    print(group)
    print()

# We can compute a dict with the groupings and use this:
groupdict = dict(list(frame1_1.groupby('key1')))
print(groupdict)
print()

# We can also change the axis of operation with axis=

# ******************** Selecting Columns **********************

# We can select directly out of the groupby object as a shorthand
# for selecting properly (i.e. before we create it)

# For large datasets, it's a good idea to select before calculating:
# i.e. df.groupby(['key1', 'key2'])['data2'].mean()

# *************** Grouping with Dicts and Series **************

frame1_7 = DataFrame(np.random.randn(5,5),
                     columns=list('abcde'),
                     index=['Joe', 'Steve', 'Wes', 'Jim', 'Travis'])
frame1_7.loc['Wes'][['b', 'c']] = np.nan
printFrame(1,7)

# Now, I have particular columns that map to colour, and I want to sum by the colours
mapping = {'a': 'red', 'b': 'red', 'c': 'blue', 'd': 'blue', 'e': 'red', 'f': 'orange'}

# We can pass the dict to groupby, and it'll do the rest for us:
frame1_8 = frame1_7.groupby(mapping, axis=1).sum()
printFrame(1,8)

# This can also be done with Series
map_series=Series(mapping)
print(map_series)
print()

frame1_8 = frame1_7.groupby(map_series, axis=1).sum()
printFrame(1,8)

# ****************** Grouping with Functions ******************

# We can also group using functions
frame1_9 = frame1_7.groupby(len).sum()
printFrame(1,9)

# We can mix and match the above things too, as they get converted 
# to arrays internally

# **************** Grouping with Index Levels *****************

# The level= argument allows us to aggregate with multiIndexes

columns = pd.MultiIndex.from_arrays([
    ['US','US','US','JP','JP'],
    [1,3,5,1,3]], names=['cty', 'tenor'])
frame1_10 = DataFrame(np.random.randn(4,5), columns=columns)
printFrame(1,10)
frame1_11 = frame1_10.groupby(level='cty', axis=1).count()
printFrame(1,11)

# *************************************************************
# ********************* Data Aggregation **********************

# We've used multiple aggregating functions so far, such as 
# sum or mean

# We can take any function which aggregates an array, and pass it 
# to the agg method

frame2_1 = frame1_1.copy()
printFrame(2,1)

def peak_to_peak(arr):
    return arr.max() - arr.min()

cols = ['key1', 'data1', 'data2']
frame2_2 = frame2_1[cols].groupby('key1').agg(peak_to_peak)
printFrame(2,2)

# Methods like quantiles and describe also work with grouping, 
# though not strictly being aggregations

# Optimised GroupBy Methods
# Method       Description
# count        Number of non-NA values
# sum          Sum of non-NA values
# mean         Mean of non-NA values
# median       Median of non-NA values
# std, var     Deviation and variance
# min, max     Min and max of non-NA values
# prod         Product of non-NA values
# first, last  First and last of non-NA values

# ****** Column-wise and Multiple Function Application ********

frame2_3 = pd.read_csv('./examples/tips.csv')

# Add a tip pct column first:
frame2_3['tip_pct'] = frame2_3['tip'] / frame2_3['total_bill']
printFrame(2,3)

# Ok, now let's group the data by meal and smoker and see means
# (this time by passing mean to the agg function):
frame2_4 = frame2_3.groupby(['time', 'smoker']).agg('mean')['tip_pct']
printFrame(2,4)

# If we pass a list of functions, we'll get each of these
frame2_5 = frame2_3.groupby(['time', 'smoker']).agg(['mean', 'std', peak_to_peak])['tip_pct']
printFrame(2,5)

# We can rename our new columns by passing tuples
frame2_5 = frame2_3.groupby(['time', 'smoker']).agg(['mean', 'std', ('gap', peak_to_peak)])['tip_pct']
printFrame(2,5)

# We can pass multiple columns to get results from them:
frame2_6 = frame2_3.groupby(['time', 'smoker']).agg(['mean', 'std', ('gap', peak_to_peak)])[['tip_pct', 'total_bill']]
printFrame(2,6)

# Finally, if we really want to, we can pass a dict for which functions to apply where, so we can
# apply different functions to different columns:
frame2_7 = frame2_3.groupby(['time', 'smoker']).agg({'tip_pct':['mean', 'std', ('gap', peak_to_peak)], 'total_bill': 'sum'})
printFrame(2,7)

# ******* Returning Aggregated Data in Unindexed Form *********

# In all of our examples, our by-variables have been returned as the index
# To prevent this, use as_index=False in the groupby

frame2_8 = frame2_3.groupby(['time', 'smoker'], as_index=False).agg({'tip_pct':['mean', 'std', ('gap', peak_to_peak)], 'total_bill': 'sum'})
printFrame(2,8)