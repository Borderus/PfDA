# Import libraries

from pandas import Series, DataFrame
from pandas_datareader import data

import pandas as pd
import numpy as np

# *************************************************************

# Define functions to print Series and Frames

def printSeries(num1, num2):
    exec(f'print("Series{num1}.{num2}:", "", series{num1}_{num2}, sep="""\n""")')
    print()

def printFrame(num1, num2):
    exec(f'print("DataFrame{num1}.{num2}:", "", frame{num1}_{num2}, sep="""\n""")')
    print()

# *************************************************************

# ************** Intro to Pandas data structures **************

# Series

# A series is a 1D array of data along with an associated array of data labels, called its index

series1_1 = Series([4, 7, -5, 3])
printSeries(1,1)

# We can see that when we assign no labels numeric values get assigned instead.
# We can get the data and index separately using the values and index objects

print(series1_1.values)
print(series1_1.index)

print()

# Let's now add some labels

series1_2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
printSeries(1,2)

# We can then select items using this index

print(series1_2['d'], "\n")

# Numpy array operations such as filtering, scalar multiplication and applying functions maintain the labels

series1_3 = series1_2[series1_2 > 0]
printSeries(1,3)
series1_4 = series1_2*2
printSeries(1,4)
series1_5 = np.exp(series1_2)
printSeries(1,5)

# We can create Series from dicts:

dict1_6 = {"Oklahoma": 3500, "Texas": 2500, "Oregon": 16000, "Utah": 5000}
series1_6 = Series(dict1_6)
printSeries(1,6)

# What happens if we use a dict and pass a contradictory index?

labels1_7 = ["California", "Texas", "Oregon", "Utah"]
series1_7 = Series(dict1_6,index=labels1_7)
printSeries(1,7)

# The value for California is NaN i.e. missing! These can be detected with the pd.isnull() and pd.notnull() methods

# Missing values add to become further missing values:

series1_8 = series1_7 + series1_6
printSeries(1,8)

# We can add a label to the series and to the index:

series1_9 = series1_7
series1_9.name = "Population"
series1_9.index.name = "State"
printSeries(1,9)

# *************************************************************

# DataFrames

frame2_1 = DataFrame({'state': ['Ohio', 'ohio', 'Ohio', 'Nevada', 'Nevada']
          , 'year': [2000, 2001, 2002, 2001, 2002]
          , 'pop': [1.5, 1.7, 3.6, 2.4, 2.9]
            })
printFrame(2,1)

# We can specify the columns and their order

frame2_2 = DataFrame(frame2_1, columns=['year', 'state', 'pop', 'debt'])
printFrame(2,2)

# We can also specify an index

frame2_3 = DataFrame(frame2_1, columns=['year', 'state', 'pop', 'debt'])
frame2_3.index=['one', 'two', 'three', 'four', 'five']
printFrame(2,3)

print("Frame2.3 States:")
print(frame2_3['state'])
print()
print("Frame2.3 Years:")
print(frame2_3.year)
print()
print("Frame2.3 Row 1: - Found by Index")
print(frame2_3.loc['one'])
print()
print("Frame2.3 Row 1: - Found by Row No:")
print(frame2_3.iloc[0])
print()

# Columns can be modified by assignment:

# Copy method to alter 1_4 and not 1_3
frame2_4 = frame2_3.copy()
frame2_4['debt'] = 16.5
printFrame(2,4)

# We cna assign series, though this can create holes:
frame2_5 = frame2_3.copy()
series2_5 = Series([-1.2, -1.5, -1.5], index=['two', 'four', 'five'])

frame2_5['debt'] = series2_5
printFrame(2,5)

# Assigning a nonexistent column creates it. We can remove it with the del keyword

frame2_5['eastern'] = (frame2_5.state.apply(lambda x: x.lower()) == 'ohio')
printFrame(2,5)

del frame2_5['eastern']
printFrame(2,5)

# We can convert nested dicts to DataFrames

dict2_6 = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame2_6 = DataFrame(dict2_6)
printFrame(2,6)

# And we can transpose it if we like

frame2_7 = frame2_6.T
printFrame(2,7)

# The keys in the inner dicts are unioned and sorted to get the index. This isn't the case
# if the index is explicitly identified

frame2_8 = DataFrame(dict2_6, index = [2001, 2002, 2003])
printFrame(2,8)

# Like the series, we can use the values object to get the data as a 2D ndarray

print("DataFrame 2.8 values:")
print(frame2_8.values)
print()

# *************************************************************

# Index Objects

# Indexes are immutable and can't have individual elements overwritten
# They do however have set functionality

# METHOD
# append: Concatenate with other index objects to produce a new index
# diff: Compute set difference as an index
# intersection: Compute set intersection
# union: Compute set union
# isin: Compute boolean array dependant on if each value is contained in passed collection
# delete: Compute new index with element at index i deleted
# drop: Compute new index by deleting passed values
# insert: Compute new index by inserting element at index i
# is_monotonic: Returns true if each element is greater than previous element
# is_unique: Returns true if each element has no duplicate values
# unique: Compute the array of unique values in the index

# e.g.
print('Duplicate check on index for DataFrame 2.6:')
print(frame2_6.index.is_unique)
print()

# *************************************************************

# ****************** ESSENTIAL FUNCTIONALITY ******************

# Reindexing

# Reindexing is - as you'd suspect - redefining an index
series3_1 = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
printSeries(3,1)
series3_2 = series3_1.reindex(['a', 'b', 'c', 'd', 'e'])
printSeries(3,2)
# As you can see, this allows us to resort the data and to redefine the domain

# We may also not want new values to be missing:
series3_3 = series3_1.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)
printSeries(3,3)

# We may also want to perform some interpolation - the method= option allows for this.
# For example, ffill forward-fills the values
series3_4 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
printSeries(3,4)
series3_5 = series3_4.reindex(range(6), method='ffill')
printSeries(3,5)
# ffill = forward fill, and bfill = back fill

# With DataFrames, reindex can be used to change either the column index or row index
frame3_1 = DataFrame(np.arange(9).reshape((3, 3)), index=['a', 'c', 'd'], columns=['Ohio', 'Texas', 'California'])
printFrame(3,1)
frame3_2 = frame3_1.reindex(['a', 'b', 'c', 'd'])
printFrame(3,2)
frame3_3 = frame3_2.reindex(columns=['Texas', 'Utah', 'California'])
printFrame(3,3)
# Both can be done at once, though only row interpolation will be done (though it'll 
# error for string indexes - do it as a method in this scenario)
frame3_4 = frame3_1.reindex(['a', 'b', 'c', 'd'], columns=['Texas', 'Utah', 'California']).ffill()
printFrame(3,4)
# Finally, reindex has a copy= argument - True by default.

# Dropping entries from an axis

frame3_5 = frame3_4.drop('d')
printFrame(3,5)
frame3_5 = frame3_5.drop(['a', 'b'])
printFrame(3,5)
# And columns can be dropped too
frame3_5 = frame3_5.drop(['California', 'Utah'], axis=1)
printFrame(3,5)

#  Indexes, selection and filtering
series3_6 = Series(np.arange(5.), index=['a', 'b', 'c', 'd', 'e'])
printSeries(3,6)
series3_7 = series3_6['b']
printSeries(3,7)
series3_8 = series3_6[1]
printSeries(3,8)
series3_9 = series3_6[2:4]
printSeries(3,9)
series3_10 = series3_6[series3_6 < 2]
printSeries(3,10)
# We can slice with labels - we just have to remember that unlike the usual approach,
# this one is endpoint-inclusive
series3_11 = series3_6['b':'d']
printSeries(3,11)
# We can alter values based on these:
series3_11['b':'c'] = 5
printSeries(3,11)
# And now: frames
frame3_6 = DataFrame(np.arange(16).reshape((4,4)), index=['Ohio', 'Colorado', 'Utah', 'New York'], columns=['one', 'two', 'three', 'four'])
printFrame(3,6)
frame3_7 = frame3_6['two']
printFrame(3,7)
frame3_8 = frame3_6[:2]
printFrame(3,8)
frame3_9 = frame3_6[frame3_6['three'] > 5]
printFrame(3,9)
# Creation of Boolean arrays
frame3_10 = (frame3_6 < 5)
printFrame(3,10)
# And now assignment. This however generates a warning saying you should use the loc function
frame3_9[frame3_9 < 8] = 0
printFrame(3,9)
# Speaking of the devil, let's use the iloc and loc method
# iloc goes for the numeric location:
frame3_11 = frame3_6.iloc[0]
printFrame(3,11)
frame3_11 = frame3_6.iloc[:, 0]
printFrame(3,11)
# loc goes for the index value
frame3_12 = frame3_6.loc['Ohio']
printFrame(3,12)
frame3_12 = frame3_6.loc[:, 'one']
printFrame(3,12)
frame3_12 = frame3_6.loc[['Ohio', 'Colorado'], 'one']
printFrame(3,12)
# And trying with booleans:
frame3_12 = frame3_6.loc[frame3_6.one < 6, 'two':'four']
printFrame(3,12)

# Arithmetic and data alignment

series3_12 = Series([7.3, -2.5, 3.4, 1.5], index=['a', 'c', 'd', 'e'])
printSeries(3,12)
series3_13 = Series([-2.1, 3.6, -1.5, 4, 3.1], index=['a', 'c', 'e', 'f', 'g'])
printSeries(3,13)
series3_14 = series3_13 + series3_12
printSeries(3,14)
# This doesn't work, with missings cropping up everywhere - and the same applies
# to DataFrames. We have to use an add method so we can define a fill value
series3_14 = series3_13.add(series3_12, fill_value=0)
printSeries(3,14)

# The same applies to DataFrames
frame3_13 = DataFrame(np.arange(12.).reshape(3,4), columns=list('abcd'))
printFrame(3,13)
frame3_14 = DataFrame(np.arange(20.).reshape(4,5), columns=list('abcde'))
printFrame(3,14)
frame3_15 = frame3_14 + frame3_13
printFrame(3,15)
frame3_15 = frame3_14.add(frame3_13, fill_value=0)
printFrame(3,15)
# The respective methods are add, sub, div, mul

# Operations between DataFrames and series

# Operations between DataFRames and Series are well-defined. Consider the interaction 
# between an ndarray and one of its rows

array3_1 = np.arange(12.).reshape((3,4))
print("Array 3.1:")
print(array3_1)
print("Array 3.1 with its first row taken away:")
print(array3_1 - array3_1[0])
# This is referred to as broadcasting. DataFrames and Series are similar:
frame3_16 = DataFrame(np.arange(12.).reshape((4,3)), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
printFrame(3,16)
series3_16 = frame3_16.iloc[0]
printSeries(3,16)
frame3_17 = frame3_16 - series3_16
printFrame(3,17)
# Now, if the indexes don't match:
series3_18 = Series(np.arange(3), index=list("bef"))
printSeries(3,18)
frame3_18 = frame3_16 - series3_18
printFrame(3,18)
# Which is crap - we need to use the arithmetic methods. Need to specify the axis to match on when we do this now:
# BB: Kept failing when the index wasn't a perfect match. Just ended up aligning them.
series3_18 = series3_18.reindex(frame3_16.columns).fillna(0)
printSeries(3,18)
frame3_18 = frame3_16.sub(series3_18, axis=1)
printFrame(3,18)

# Function application and mapping

# NumPy ufuncs work fine:
frame3_19 = DataFrame(np.random.randn(4,3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
printFrame(3,19)
frame3_20 = abs(frame3_19)
printFrame(3,20)
# We can also create a lambda function and then pass it to the apply method
# This allows us to apply the function to each column or row as if it were a 1D array
f = lambda x: x.max() - x.min()
frame3_21 = frame3_19.apply(f)
printFrame(3,21)
frame3_22 = frame3_19.apply(f, axis=1)
printFrame(3,22)
# The function applied doesn't have to return a scalar value, it can also return a series:
def f(x):
    return Series([x.min(), x.max()], index=['min', 'max'])

frame3_23 = frame3_19.apply(f)
printFrame(3,23)
# The applymap function will allow us to apply a function to individual elements
format = lambda x: '%.2f' % x
frame3_24 = frame3_19.applymap(format)
printFrame(3,24)
# Key points - apply for rows/columns, applymap for elements

# Sorting and Ranking

# Series can be sorted with the sort_index function:
series3_24 = Series(np.arange(4), index=list('dabc'))
printSeries(3,24)
series3_24 = series3_24.sort_index()
printSeries(3,24)
# Frames can be sorted similarly, with the axis specified for column sorting
frame3_25 = DataFrame(np.arange(8).reshape((2,4)), index=['three', 'one'], columns=list('dabc'))
printFrame(3,25)
frame3_26 = frame3_25.sort_index()
printFrame(3,26)
frame3_27 = frame3_25.sort_index(axis=1)
printFrame(3,27)
# We can perform a descending sort
frame3_28 = frame3_25.sort_index(axis=1, ascending=False)
printFrame(3,28)
# If we want to sort arrays by values, use the sort_values method. Missing values go to the end by default
series3_25 = series3_24.sort_values()
printSeries(3,25)
# Ordering dataframes we use the sort_values method
frame3_29 = frame3_25.sort_values(by=['d', 'a'])
printFrame(3,29)
# Ranking allows us to assign a number. Matching results get the average position by default 
# (cat and dog are matching for 2nd and 3rd positions, overall 2.5)
series3_30 = Series({"cat": 4, "penguin": 2, "dog": 4, "spider": 8, "snake": np.nan})
printSeries(3,30)
series3_31 = series3_30.rank()
printSeries(3,31)
# We can alter the rules applied to tiebreak a match by picking the first
series3_31 = series3_30.rank(method="first")
printSeries(3,31)
# We can award them the higher possible position (or max for lower)
series3_31 = series3_30.rank(method="min")
printSeries(3,31)

# Axis indexes with duplicate values

series3_32 = Series(range(5), index=['a', 'a', 'b', 'b', 'c'])
printSeries(3,32)
print("This series' index's is_unique property resolves to", series3_32.index.is_unique)
# Data selection changes here - selecting a value with dupes returns a series rather than a scalar
print(series3_32['c'])
print(series3_32['a'])

# *************************************************************

# Summarising and Computing Descriptive Statistics

# We can sum with the sum function:
frame4_1 = DataFrame([[1.4, np.nan], [7.1, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=list('abcd'), columns=['one', 'two'])
printFrame(4,1)
frame4_2 = frame4_1.sum()
printFrame(4,2)
frame4_3 = frame4_1.sum(axis=1)
printFrame(4,3)
# By default this skips missing values - we can stop this behaviour
frame4_4 = frame4_1.sum(axis=1, skipna=False)
printFrame(4,4)
# Some methods return the index value where the maximum or minimum are stored
frame4_5 = frame4_1.idxmax()
printFrame(4,5)
# Some methods are accumulations:
frame4_6 = frame4_1.fillna(0).cumsum()
printFrame(4,6)
# Or we can just use the describe method (which works like proc freq/means in SAS depending on data type)
frame4_7 = frame4_1.describe()
printFrame(4,7)
# Some summary statistics, like correlation and covariance, are computed from pairs of arguments.

# Unique Values, Value Counts and Membership

series4_8 = Series(list('cadaabbcc'))
printSeries(4,8)
# The unique method gives us the deduped list of values
series4_9 = series4_8.unique()
printSeries(4,9)
# Get frequencies
series4_10 = series4_8.value_counts()
printSeries(4,10)
# Get description
series4_11 = series4_8.describe()
printSeries(4,11)
# Get a Boolean based on membership of another list
series4_12 = series4_8.isin(['b','c'])
printSeries(4,12)
# We may want data froma Frame based on number of occurrences for a histogram - in this case we could use:
frame4_13 = DataFrame({'Q1': [1, 3, 4, 3, 4],
                       'Q2': [2, 3, 1, 2, 3],
                       'Q3': [1, 5, 2, 4, 4]})
printFrame(4,13)
frame4_14 = frame4_13.apply(pd.value_counts).fillna(0)
printFrame(4,14)

# *************************************************************

# Handling Missing Data

# Checking for null values
series5_1 = Series(['aardvark', 'artichoke', np.nan, 'avocado'])
printSeries(5,1)
print(series5_1.isnull(),'')

# Filtering them out

series5_2 = series5_1.dropna()
print(series5_2)
# Frames are more complex - dropna by default removes all rows and columns containing any missing values
frame5_3 = DataFrame([[1., 6.5, 3.], [1., np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, 6.5, 3.]])
printFrame(5,3)
frame5_4 = frame5_3.dropna()
printFrame(5,4)
# The how argument is how we alter this behaviour. COlumns can be done with the axis argument
frame5_5 = frame5_3.dropna(how='all')
printFrame(5,5)
# We can also specify a threshold to stick to - a row that has at least this many NaN will be dropped
frame5_6 = frame5_3.dropna(thresh=2)
printFrame(5,6)

# Or we may want to fill them in:

frame5_7 = frame5_3.fillna(0)
printFrame(5,7)
# If we invoke it with a dict we can decide what to do with each column:
frame5_8 = frame5_3.fillna({0:-9, 2:7})
printFrame(5,8)
# The inplace option can be used to not create a copy, and method= exists like it did with reindexing

# *************************************************************

# Hierarchical Indexing

# Take a look at this and you'll get an idea
series6_1 = Series(np.random.randn(10), index=[['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd'], [1, 2, 3, 1, 2, 3, 1, 2, 2, 3]])
printSeries(6,1)
# This is a MultiIndex, and it lets us do things like partial indexing
series6_2 = series6_1['b':'c']
printSeries(6,2)
series6_2 = series6_1.loc['b':'c']
printSeries(6,2)
series6_3 = series6_1.loc[:, 2]
printSeries(6,3)
# Hierarchical indexing plays a critical role in reshaping data and group-based
# operations. We can convert between these and DataFrames like so:
frame6_4 = series6_1.unstack()
printFrame(6,4)
series6_5 = frame6_4.stack()
printSeries(6,5)
# With a DataFrame, either axis can have a hierarchical index:
frame6_6 = DataFrame(np.arange(12).reshape((4,3)), index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]], columns=[['Ohio', 'Ohio', 'Colorado'], ['Green', 'Red', 'Green']])
printFrame(6,6)
series6_6 = frame6_6.stack()
printSeries(6,6)
# We can name the axes:
frame6_6.index.names = ['key1', 'key2']
frame6_6.columns.names = ['state', 'colour']
printFrame(6,6)
# Groups of columns can also be selected through partial indexing:
frame6_7 = frame6_6['Ohio']
printFrame(6,7)

# Reordering and sorting levels

# We can swap the levels with swaplevel
frame6_8 = frame6_6.swaplevel('key1', 'key2')
printFrame(6,8)
# We can sort by a particular level using sort_values
frame6_8 = frame6_8.sort_values(by='key2')
printFrame(6,8)

# Summary Statistics

# Most summary statistics support a levels argument
frame6_9 = frame6_8.sum(level='colour', axis=1)
printFrame(6,9)
# Under the bonnet, this is just using the groupby method

# Using a dataframe's columns

# We may want to bounce between one and the other: 
frame6_10 = DataFrame({'a': range(7), 'b': range(7, 0, -1), 'c': ['one', 'one', 'one', 'two', 'two', 'two', 'two'], 'd': [0, 1, 2, 0, 1, 2, 3]})
frame6_10 = frame6_10.set_index(['c', 'd'])
printFrame(6,10)
frame6_10 = frame6_10.reset_index()
printFrame(6,10)

# *************************************************************

# Other Pandas topics

# Integer Indexing:
# If an index is composed of integers, then querying it for a number will default to labels. Use iloc to get positions

# Panel Data:
# A panel is effective a 3D DataFrame
# Create one using the Panel function and pass is a dictionary of DataFrames
# Can just use hierarchical indexing instead
# to_frame and to_panel methods to stack and unstack
# Things otherwise basically work as expected