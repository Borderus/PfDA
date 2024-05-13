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

# ****** Group-wise Operations and Transformations ********

# Aggregation is not the only group-level action available to us.

frame3_1 = frame1_1.copy()
printFrame(3,1)

# Let's say we want to define a column with the mean of each group. Then:

frame3_2_means = frame3_1.groupby('key1').mean().add_prefix('mean_')
frame3_2 = pd.merge(frame3_1, frame3_2_means, left_on='key1', right_index=True)

printFrame(3,2)

# This works, but is cumbersome. The transform function lets us keep all records while
# applying a function to a group

frame3_2 = frame3_1.copy()
frame3_2[['mean_data1', 'mean_data2']] = frame3_1.groupby('key1')[['data1', 'data2']].transform(np.mean)
printFrame(3,2)

# We could also use it to get how far each record is from the mean of its group

def demean(arr):
    return arr - arr.mean()

frame3_2[['mean_dif']] = frame3_2.groupby('key1')[['data1']].transform(demean)
printFrame(3,2)

# This method is also quite rigid though. For a most general tool, we use apply
# Apply splits the object being manipulated into pieces, invokes the function on
# each piece and attempts to concatenate the pieces together.

# Going back to our old dataset of smokers:
frame3_3 = frame2_3.copy()
printFrame(3,3)

# Let's first select the top 5 percent values by group.

def top(df, n=5, column='tip_pct'):
    return df.sort_values(column)[-n:]

frame3_4 = top(frame3_3)
printFrame(3,4)

# We've now pulled the top 5 tippers by percentage
# Let's have a look at for smokers and non-smokers

frame3_5 = frame3_3.groupby('smoker').apply(top)
printFrame(3,5)

# To pass keywords to the function, add extra arguments

frame3_6 = frame3_3.groupby(['smoker', 'day']).apply(top, n=1)
printFrame(3,6)

# We can, if we want, call methods like describe on this:
f = lambda x: x.describe()
frame3_7 = frame3_3.groupby('smoker').apply(f)
printFrame(3,7)

# And as before, we can pass group_keys=False to the groupby() to not
# have those values in the index

# *** Quantile and Bucket Analysis ***

# We can combine quantile tools with groupby to create powerful methods of examining 
# populations

frame3_8 = DataFrame({'data1':np.random.randn(1000),
                   'data2':np.random.randn(1000)})
printFrame(3,8)
factor = pd.cut(frame3_8['data1'], 4)
print(factor[:10])
print()

# We can then group by this factor to run a function on it
# In this case we'll determine some useful stats for each record

def get_stats(group):
    return {'min': group.min(), 'max': group.max(),
            'count': group.count(), 'mean': group.mean()}

frame3_9 = frame3_8['data2'].groupby(factor).apply(get_stats).unstack()
printFrame(3,9)

# These were equal-length buckets - to get equal quantity you'd use qcut.

# *** Example: Filling missing values with group-specific values ***

# When cleaning up data, sometimes you'd want to filter out missing values.
# But in other, you may want to impute these (i.e. interpolate), possibly with the mean of a group
# We can do this with the fillna and apply methods

states=['Ohio', 'New York', 'Vermont', 'Florida',
        'Oregon', 'Nevada', 'California', 'Idaho']

group_key = ['East']*4 + ['West']*4

series3_10 = Series(np.random.randn(8), index=states)
series3_10[['Vermont', 'Nevada', 'Idaho']] = np.nan

printSeries(3,10)

# We will now assign the missing values the mean of their group
fill_mean = lambda x: x.fillna(x.mean())
series3_11 = series3_10.groupby(group_key).apply(fill_mean)
printSeries(3,11)

# Alternately, we can give them pre-assigned values like this
fill_values = {'East': 0.5, 'West': -1}
fill_func = lambda x: x.fillna(fill_values[x.name])
series3_12 = series3_10.groupby(group_key).apply(fill_func)
printSeries(3,12)

# *** Examples: Shuffling a deck ***

suits = ['H', 'S', 'C', 'D']
card_val = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
base_names = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K']
cards = []

for suit in suits:
  for num in base_names:
    cards.append(str(num)+suit)

series3_13 = Series(card_val, index=cards)
printSeries(3,13)

# Now, to shuffle the deck, we can use the random permutation method:
def draw(deck, n=5):
    return deck.take(np.random.permutation(len(deck))[:n])
# Take method allowing selection of the nth element

print(draw(series3_13))

# Now let's get two random cards from each suit
get_suit = lambda card: card[-1]
series3_14 = series3_13.groupby(get_suit).apply(draw, n=2)
print(series3_14)

# *** Example: Grouped correlations ***

# Let's get the Yahoo Finance data
frame3_15 = pd.read_csv('./examples/stock_px.csv')
printFrame(3,15)

# Now, it might be of interest to calculate percentage daily return. Let's dropna and calculate percentages
pct_chg = lambda x: x.pct_change()
frame3_16 = frame3_15.copy()
frame3_16[['AAPL', 'MSFT', 'XOM', 'SPX']] = frame3_15[['AAPL', 'MSFT', 'XOM', 'SPX']].transform(pct_chg).dropna()
printFrame(3,16)

# Define correlation and year functions
spx_corr = lambda x: x.corrwith(x['SPX'])
get_year = lambda x: x[0:4]
frame3_17 = frame3_16.copy()
frame3_17['year'] = frame3_17['Unnamed: 0'].apply(get_year)
printFrame(3,17)
frame3_18 = frame3_17.groupby(frame3_17['year']).apply(spx_corr)
printFrame(3,18)

# *** Example: Group-wise Linear Regression ***

# The statsmodels library allows drawing lines-of-best-fit with
# linear regression. Apply then allows us to do it on a 
# group-by-group scale

import statsmodels.api as sm
def regress(data, yvar, xvars):
    Y = data[yvar]
    X = data[xvars]
    X['intercept'] = 1.
    result = sm.OLS(Y, X, missing='drop').fit()
    return result.params

frame3_19 = frame3_17.groupby(frame3_17['year']).apply(regress, 'AAPL', ['SPX'])
printFrame(3,19)

# ******* Pivot Tables and Cross-Tabulation *********

frame4_1 = frame2_3.copy()
printFrame(4,1)

# We can produce a pivot with the pivot_table method

frame4_2 = frame4_1.pivot_table(index=['day', 'smoker'], columns=['time'])
printFrame(4,2)

# We can get a count of the whole sample with margins=True.

# We can change the function (default mean) with aggfunc:
frame4_3 = frame4_1.pivot_table('tip', index='time', aggfunc=sum)
printFrame(4,3)

# If there are nans, you can pass fill_value=0 (or another value) to populate these

# *** Cross-tabulations ***

# Cross-tabulations are special instances of pivot tables that store frequencies.

frame4_4 = pd.read_csv('./examples/handedness.csv')
printFrame(4,4)

frame4_5 = pd.crosstab(frame4_4['gender'], frame4_4['handedness'], margins=True)
printFrame(4,5)

# ******* Example: 2012 Federal Election Commission Database *********

frame5_1 = pd.read_csv('./datasets/fec/P00000001-ALL.csv', low_memory=False)
printFrame(5,1)

party = {
    'Obama, Barack': 'DEMOCRAT', 
    'Romney, Mitt': 'REPUBLICAN'
}

frame5_1["party"] = frame5_1.loc[:,"cand_nm"].map(party)

frame5_2 = frame5_1[(frame5_1['contb_receipt_amt'] > 0) & (frame5_1['cand_nm'].isin(['Obama, Barack', 'Romney, Mitt']))]
printFrame(5,2)

# DONATION STATS BY OCCUPATION AND EMPLOYER

# We can look at what the most frequent contributors were:

series5_3 = frame5_2.loc[:,'contbr_occupation'].value_counts()[:10]
print(series5_3)

# But information requested is in here twice, under two different names. Let's map those together, as well as C.E.O -> CEO

occ_mapping = {
    "INFORMATION REQUESTED PER BEST EFFORTS": "NOT PROVIDED",
    "INFORMATION REQUESTED": "NOT PROVIDED",
    "INFORMATION REQUESTED (BEST EFFORTS)": "NOT PROVIDED",
    "C.E.O.": "CEO"
}

# Define a function which then uses the get method to return the output of this. The get method means that if there is no
# mapping, we'll receive the original value fed in

def get_occ(x):
    return occ_mapping.get(x, x)

frame5_2["contbr_occupation"] = frame5_2.loc[:,"contbr_occupation"].map(get_occ)

series5_3 = frame5_2.loc[:,'contbr_occupation'].copy().value_counts()[:10]
print(series5_3)

# Now, let's use a pivot table to get the contribution amount by party and occupation
frame5_4 = frame5_2.pivot_table("contb_receipt_amt", index='contbr_occupation', columns='party', aggfunc='sum')
frame5_4 = frame5_4[frame5_4.sum(axis='columns') > 2000000]
printFrame(5,4)

# frame5_4.plot(kind='barh')
# plt.show()

# BUCKETING DONATION AMOUNTS

# Let's look at size on contributions

bins = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
labels = pd.cut(frame5_2["contb_receipt_amt"], bins)
frame5_5 = frame5_2.groupby(["cand_nm", labels]).size().unstack(level=0)
printFrame(5,5)

frame5_6 = frame5_2.groupby(["cand_nm", labels])["contb_receipt_amt"].sum().unstack(level=0)
printFrame(5,6)

# fig, ax= plt.subplots(2)
# frame5_5.plot(kind='bar', ax=ax[0])
# frame5_6.plot(kind='bar', ax=ax[1])
# plt.show()

# DONATION STATS BY STATE

frame5_7 = frame5_2.groupby(["cand_nm", "contbr_st"])["contb_receipt_amt"].sum().unstack(level=0).fillna(0)
frame5_7 = frame5_7[frame5_7.sum(axis="columns") > 100000]
printFrame(5,7)