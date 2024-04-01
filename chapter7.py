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

# ********** Data Loading, Storage and File Formats ***********

# Merging

frame1_1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'a', 'b'], 'data1': range(7)})
printFrame(1,1)
frame1_2 = DataFrame({'key': ['a', 'b', 'd'], 'data2': range(3)})
printFrame(1,2)
# We cna merge these with the .merge() method - in this instance will get a many-to-one
# Defaults to an inner join
frame1_3 = pd.merge(frame1_1, frame1_2)
printFrame(1,3)
# Best practice to explicitly state by-variable (otherwise will use all overlapping cols)
frame1_3 = pd.merge(frame1_1, frame1_2, on='key')
printFrame(1,3)
# If the by-variables have different names in each table, can specify each
frame1_4 = DataFrame({'newkey': ['a', 'b', 'd'], 'data2': range(3)})
printFrame(1,4)
frame1_5 = pd.merge(frame1_1, frame1_4, left_on='key', right_on='newkey')
printFrame(1,5)
# We can switch between inner, left, right and outer joins with the how= arg. Cross is also
# a valid option, for a Cartesian join
frame1_6 = pd.merge(frame1_1, frame1_2, on='key', how='left')
printFrame(1,6)
# To merge with multiple keys, pass a list of them
frame1_7 = DataFrame({'key1': ['foo', 'foo', 'bar'],
                       'key2': ['one', 'two', 'one'],
                       'lval': [1, 2, 3]})
printFrame(1,7)
frame1_8 = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                       'key2': ['one', 'one', 'one', 'two'],
                       'lval': [4, 5, 6, 7]})
printFrame(1,8)
frame1_9 = pd.merge(frame1_7, frame1_8, on=['key1', 'key2'], how='outer')
printFrame(1,9)
# When column names accidentally align, we can rename or merge has the suffixes option
# for specifying strings to append to overlapping names
frame1_10 = pd.merge(frame1_7, frame1_8, on='key1', suffixes=('_left', '_right'))
printFrame(1,10)

# OPTIONS FOR MERGE:

# how: Inner, left, right, outer or cross (i.e. cartesian)
# on: by-variable for join
# left_on: specify 'on' for left table
# right_on: specify 'on' for right table
# left_index: use row index in left as join key
# right_index: use row index in right as join key
# sort: sort data lexicographically by join keys - default to True
# suffixes: specify suffixes to add to overlapping column names - defaults to ('_x', '_y')
# copy: If false, avoid copying data into resulting data structure in some exceptional cases.
#       Always copies by default

# Merging on indexes

frame1_11 = DataFrame({'key': ['a', 'b', 'a', 'a', 'b', 'c'], 'value': range(6)})
printFrame(1,11)
frame1_12 = DataFrame({'group_val': [3.5, 7]}, index= ['a', 'b'])
printFrame(1,12)
frame1_13 = pd.merge(frame1_11, frame1_12, left_on='key', right_index=True)
printFrame(1,13)
# Using an outer join:
frame1_14 = pd.merge(frame1_11, frame1_12, left_on='key', right_index=True, how='outer')
printFrame(1,14)
# With hierarchically-indexed data, things get slightly more interesting:
frame1_15 = DataFrame({'key1': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
                       'key2': [2000, 2001, 2002, 2001, 2002],
                       'data': np.arange(5)})
printFrame(1,15)
frame1_16 = DataFrame(np.arange(12).reshape((6,2)),
                       index=[['Nevada', 'Nevada', 'Ohio', 'Ohio', 'Ohio', 'Ohio'],
                              [2001, 2000, 2000, 2000, 2001, 2002]],
                       columns=['event1', 'event2'])
printFrame(1,16)
# We have to pass a list of left keys in this scenario
frame1_17 = pd.merge(frame1_15, frame1_16, left_on=['key1', 'key2'], right_index=True, how='outer')
printFrame(1,17)
# We can also merge things by using the indexes from both sides of the merge
frame1_18 = DataFrame(np.arange(6).reshape((3,2)), index=['a', 'c', 'e'], columns=['Ohio', 'Nevada'])
printFrame(1,18)
frame1_19 = DataFrame(np.arange(7, 15).reshape((4,2)), index=['b', 'c', 'd', 'e'], columns=['Missouri', 'Alabama'])
printFrame(1,19)
frame1_20 = pd.merge(frame1_18, frame1_19, how='outer',  left_index=True, right_index=True)
printFrame(1,20)
# There exists a convenient join method that we will opt not to use

# Concatenation

# Start with NumPy's concatenation method for arrays
arr = np.arange(12).reshape((3,4))
print("Original array:")
print(arr)
print("Vertical concatenation:")
print(np.concatenate([arr,arr], axis=0))
print("Horizontal concatenation:")
print(np.concatenate([arr,arr], axis=1))
# Now onto Pandas and Series
series1_1 = Series(np.arange(4), index=list('abcd'))
printSeries(1,1)
series1_2 = Series(np.arange(2,5), index=list('cde'))
printSeries(1,2)
series1_3 = Series(np.arange(3,7), index=list('defg'))
printSeries(1,3)
series1_4 = pd.concat([series1_1, series1_2, series1_3])
printSeries(1,4)
# By changing the axis we can create a dataframe
series1_5 = pd.concat([series1_1, series1_2, series1_3], axis=1)
printSeries(1,5)
# And we can restrict it with the join argument
# By changing the axis we can create a dataframe
series1_6 = pd.concat([series1_1, series1_2, series1_3], axis=1, join='inner')
printSeries(1,6)
# We can use the keys argument to make it hierarchical on source
series1_7 = pd.concat([series1_1, series1_1, series1_3], keys=['one', 'two', 'three'])
printSeries(1,7)
# And use unstack to convert into a frame:
frame1_20 = series1_7.unstack()
printFrame(1,20)
# If we combine the series horizontally, then the keys become the column headers
frame1_21 = pd.concat([series1_1, series1_1, series1_3], axis=1, keys=['one', 'two', 'three'])
printFrame(1,21)
# Similar logic applies to dataframes. If we pass a dict of options rather than a list, this
# will be equivalent to the keys option:
frame1_22 = pd.concat({'one': series1_1, 'two': series1_1, 'three': series1_3}, axis=1)
printFrame(1,22)
# Finally, when we don't care about the row index, we can pass ignore_index=True
frame1_23 = DataFrame(np.random.randn(3,4), columns=list('abcd'))
printFrame(1,23)
frame1_24 = DataFrame(np.random.randn(2,3), columns=list('bda'))
printFrame(1,24)
frame1_25 = pd.concat([frame1_23, frame1_24], ignore_index=True)
printFrame(1,25)

# CONCAT FUNCTION ARGUMENTS:

# axis: axis to concatenate along
# join: outer or inner
# keys: Values to associate with objects being concatenated - will form hierarchical index
#       if labels already present
# levels: Specific indexes to use a hierarchical index level or levels if key passed
# names: names for aforementioned hierarchical levels
# verify_integrity: Check newly created axis for dupes and raise exception if so
# ignore_index: Do not preserve indexes

# Combining data with overlap

# This section is about the combine_first method, which is just a coalesce

series1_8 = Series([np.NaN, 2.5, np.NaN, 3.5, 4.5, np.NaN], index=list('fedcba'))
printSeries(1,8)
series1_9 = Series([5, 4, 3, 2, 1, np.NaN], index=list('fedcba'))
printSeries(1,9)
series1_10 = series1_8.combine_first(series1_9)
printSeries(1,10)

# *************************************************************

# Reshaping and Pivoting

# Reshaping with hierarchical indexing

# Hierarchical indexing provides a consistent wat ro rearrange data in a DataFrame
# As we've seen before, stack() and unstack() are the main methods for this
# We can also specify the level that is stacked or unstacked by passing a number or the 
# index name to the method.

# Pivoting long to wide format

# A common way to store time series is in a long or stacked format, like so:
frame2_1 = pd.read_table("examples/ldata.txt", sep="\s+")
printFrame(2, 1)
# We often store this data like this in things like SQL schemas and other relational databases,
# as the introduction of a new item type doesn't cause issues here.
# However, as date and item together here form a primary key, it's less straightforward to use
# a compound primary key like that - so, we can just create columns for each item type and 
# have the date as the sole primary key. This is called pivoting:
frame2_2 = frame2_1.pivot('date', 'item', 'value')
printFrame(2, 2)
# Of course, if we don't specify the values we want in the table, it becomes clear that this is effectively
# stacking and hierarchical indexing
frame2_3 = frame2_1.pivot('date', 'item')
printFrame(2, 3)

# *************************************************************

# Data Transformation

# Removing duplicates

frame3_1 = DataFrame({'k1':['one']*3+['two']*4, 'k2': [1, 1, 2, 3, 3, 4, 4]})
printFrame(3,1)

# The .duplicated() method returns a boolean on whether a row is a duplicate of a previous value 
#   (i.e. False for first occurrence, True for second).
# We can dedupe with the .drop_duplicates() method

frame3_2 = frame3_1.drop_duplicates()
printFrame(3,2)

# We would specify duplicates in a particular column by passing the column name to the method

frame3_3 = frame3_1.drop_duplicates(['k1'])
printFrame(3,3)

# We can also take the last rather than first value:

frame3_4 = frame3_1.drop_duplicates(['k1'], keep='last')
printFrame(3,4)

# n.b. This can also be set to False to remove all duplicated values

# ----

# Transforming data via a function or mapping

# We can transform a dataframe on a value-by-value basis using the map method

frame3_5 = DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami', 'corned beef',
                           'Bacon', 'pastrami', 'honey ham', 'nova lox', 'gravlax'],
                           'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6, 5.5]})
printFrame(3,5)

meat_to_animal = {'bacon': 'pig',
                  'pulled pork': 'pig',
                  'pastrami': 'cow',
                  'corned beef': 'cow',
                  'honey ham': 'pig',
                  'nova lox': 'salmon',
                  'gravlax': 'salmon'
                  }

frame3_5['animal'] = frame3_5['food'].map(str.lower).map(meat_to_animal)

printFrame(3,5)

# We could also pass a lambda function that does all the work:

frame3_5['animal2'] = frame3_5['food'].map(lambda x: meat_to_animal[x.lower()])

printFrame(3,5)

# ----

# Replacing values

# Replacing missing data with fillna can be thought of as a 
# special case of more general value replacement. In the same
# way that map allows us to modify a subset, we can also use
# replace. Consider the following series, and assume that 
# -999 is a sentinel for missing data:
series3_1 = Series([1, -999, 2, -999, -1000, 3])
printSeries(3,1)
series3_2 = series3_1.replace(-999, np.nan)
printSeries(3,2)
# Replacing more than one value:
series3_2 = series3_1.replace([-999, -1000], np.nan)
printSeries(3,2)
# Or to replace multiple with multiple values
series3_2 = series3_1.replace([-999, -1000], [np.nan, 0])
printSeries(3,2)
# This can also be a dict:
series3_2 = series3_1.replace({-999: np.nan, -1000: 0})
printSeries(3,2)

# ----

# Renaming axis indexes

frame3_6 = DataFrame(np.arange(12).reshape((3,4)),
                    index=['Ohio', 'Colorado', 'New York'],
                    columns=['one', 'two', 'three', 'four'])
printFrame(3,6)
# The axes, much like for a Series, have a map method
frame3_6.index = frame3_6.index.map(str.upper)
printFrame(3,6)
# To do this without transforming the original, we can use rename()
frame3_7 = frame3_6.rename(index=str.title, columns=str.upper)
printFrame(3,7)
# We can use rename with dicts to change col and row names
frame3_8 = frame3_7.rename(index={'Ohio': 'INDIANA'}, 
                           columns={'THREE': 'peekaboo'})
printFrame(3,8)
# Finally, to not copy, use inplace=True

# ----

# Discretisation and Binning

# Suppose we want to split people into age ranges or another similar category:
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)
# n.b. the closed end can be changed with right=False
print("Categorisation of ages into buckets:")
print(cats)
# This is a special object called a Categorical
# It can be treated like an array of strings, and has a label and level object

print("\nAge category labels:")
print(cats.categories)

print("\nAge category levels:")
print(cats.codes)

print("\nAge category counts:")
print(pd.value_counts(cats))

# We can assign labels to the data and simply specify a number of 
# buckets - cut will then slice them evenly
group_names = ['Youth', 'Young_Adult', 'Middle_Aged', 'Senior']
cats2 = pd.cut(ages, 4, labels=group_names)
print('\n Another categorisation:')
print(cats2)
# Alternately, the qcut will return roughly even bins calculated on quartiles (or whatever split you want)
cats3 = pd.qcut(ages, 4)
print('\n Quartiles:')
print(cats3)
# Or we can use custom quantiles
cats4 = pd.qcut(ages, [0, 0.1, 0.5, 0.9, 1])
print('\n Custom Quantiles:')
print(cats4, '\n')

# ----

# Detecting and Filtering Outliers

# Let's start by randomly generating some normal data
np.random.seed(12345)
frame3_9 = DataFrame(np.random.randn(1000, 4))
print('Description of Frame3.9:')
print(frame3_9.describe())
# Now say we wanted to find the cases where magnitude of column 3 exceeded
# the number 3:
frame3_10 = frame3_9[np.abs(frame3_9[3]) > 3]
printFrame(3,10)
# Or where any column does:
frame3_10 = frame3_9[(np.abs(frame3_9) > 3).any(1)]
printFrame(3,10)
# We could then cap this with something like:
#   data[np.abs(data>3)] = np.sign(data)*3

# ----

# Permutation and Random Sampling

sampler = np.random.permutation(5)
print(sampler)

# We can then apply this permutation to a dataframe:
frame3_11 = DataFrame(np.arange(5*4).reshape(5,4))
print(frame3_11.take(sampler))
# We can collect random subsets without replacement by slicing permutations:
random7_20wo = np.random.permutation(20)[:7]
print("Random sample without replacement of 7 numbers 1-20")
print(random7_20wo)
# With replacement, use randint
random7_20wo = np.random.randint(1, 20, 7)
print("Random sample with replacement of 7 numbers 1-20")
print(random7_20wo)

# ----

# Calculating Flags

frame3_12 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'], 'data1': range(6)})
printFrame(3,12)
frame3_12 = pd.get_dummies(frame3_12['key'])
printFrame(3,12)

# We can get more complex using the MovieLens dataset:
mnames = ['movie_id', 'title', 'genres']

frame3_13 = pd.read_table('datasets/movielens/movies.dat', sep='::', names=mnames)
printFrame(3,13)
frame3_14 = frame3_13[['title']].join(pd.get_dummies(frame3_13, 'genres'))
printFrame(3,14)
# But this returns a column for every combination fo genres.
# So we have to be a little clever
# Create a generator (kinda like a view) for splitting out the genres
genre_iter = (set(x.split('|')) for x in frame3_13.genres)
# Get a list of genres
genres = sorted(set.union(*genre_iter))
print(genres)
# Construct an empty dataframe to fill with our flag info
flags = DataFrame(np.zeros((len(frame3_13), len(genres))), 
                  columns = genres)
for i, gen in enumerate(frame3_13.genres):
    flags.loc[i, gen.split('|')] = 1
frame3_15 = frame3_13.join(flags)
printFrame(3,15)
print("Movie 1 Genres:")
print(frame3_15.loc[0])

# A good trick for stats is to combine get_dummies with a discretisation
#   function like cut to create flags for different quantiles.

# *************************************************************

# String Manipulation

# We create a string to play with:
val = 'a, b,    guido'
print()
print("val =", val)
# We can use the split function to break this up, and then strip to remove whitespace
pieces = [x.strip() for x in val.split(',')]
print("pieces =", pieces)
# We could then use the join function to glue these back together again
newstring = "::".join(pieces)
print("newstring =", newstring)
# We have numerous ways to search for a substring
print()
print("in keyword search:", 'guido' in newstring)
print("find method search:",  newstring.find('guido'))
print("index method search:",  newstring.index('guido'))
# Note that the difference between find and index is that index raises an exception if the string is not found
# Finally, we can count how often a substring occurs
print("count occurences of guido:",  newstring.count('guido'))
# And replace allows us to replace substrings
print("replace occurences of guido:",  newstring.replace('guido', 'inigo'))

# FUNCTION               DEFINITION
# count                  Return the number of non-overlapping occurrences of the substring within the string
# endswith, startswith   Returns True if string ends/starts with substring
# join                   Use string as delimiter for concatenating sequence of other strings
# index                  Return position of first character in substring if found. Raises ValueError if not found
# find                   Return position of first character in substring if found. Returns -1 if not found
# rfind                  Return position of first character of last occurrence in substring if found. Returns -1 if not found
# replace                Replace occurrences of string with another string
# strip, rstrip, lstrip  Trim all/trailing/leading blanks
# split                  Break string into list of strings using specified delimiter (default space)
# lower, upper           Change case to lower/uppercase
# ljust, rjust           Left-justify or right-justify

# Regex

# Python has a module called re that handles regex. This has its own split method:
pieces2 = re.split('\s+', val)
print("pieces2 =", pieces2)
# If we use a piece of regex a lot, it's an idea to compile it:
regex = re.compile('\s+')
pieces2 = regex.split(val)
print("pieces2 (with compile) =", pieces2)
# We can also get all matching substrings with 'findall'
pieces2 = regex.findall(val)
print("find all spaces:", pieces2)
# Match and search, on the other hand, looks for the first instance
# The search method returns a match object, which can then be used to determine locations
# The match method works similarly, but only looks at the start of the string
print("search for space:", regex.search(val))
print("use this to get the string back:", '"'+val[regex.search(val).start():regex.search(val).end()]+'"')
print("match method for space:", regex.match(val))
# The sub function, finally, allows for substitutions to be performed:
print("substitute spaces with exclamation marks:", regex.sub('!', val))

# Vectorised String Functions in Pandas

# FUNCTION               DEFINITION
# cat                    Concatenate strings element-wise with optional delimiter
# contains               Return boolean array based on if string contains substring/search pattern
# count                  Return the number of non-overlapping occurrences of the substring within the string
# endswith, startswith   Equivalent to x.endswith and x.statswith for each element
# findall                Compute list of all occurrences of pattern for each string
# get                    Index into each element (retrieve i-th element)
# join                   Concatenate strings based on pass separator
# len                    Compute length of each string
# lower, upper           Convert to lower/uppercase
# match                  Use re.match on each element
# pad                    Add whitespace to left, right or both sides of string
# center                 Equivalent to .pad(side='both')
# repeat                 Repeat strings
# replace                Replace occurence of search pattern
# slice                  Slice each string
# split                  Split strings based on delimiter or regex
# strip, rstrip, lstrip  Trim all/trailing/leading blanks

# *************************************************************

# Example: USDA Food Database

# Let's play with the USDA food data:
db = json.load(open('datasets/usda_food/database.json'))
print('\n'+str(len(db)))
print('\n',db[0].keys())

# Lets read our new JSON into a DataFrame:
info_keys = ['description', 'group', 'id', 'manufacturer']
frame4_1 = DataFrame(db, columns=info_keys)
printFrame(4,1)

# Let's see the distribution of food groups:
print(pd.value_counts(frame4_1['group'])[:10])

# Let's begin examining nutrients. First, let's pull the nutrient data into a DataFrame
nutrients = []

# Create a list of small dataframes, then concatenate these
for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)
# print(nutrients.iloc[0])

# Dedupe this:
nutrients = nutrients.drop_duplicates()

# Rename column names to be more descriptive:
col_mapping = {'description': 'food',
               'group': 'fgroup'
}
col_mapping2 = {'description': 'nutrient',
               'group': 'nutgroup'
}

frame4_1 = frame4_1.rename(columns=col_mapping, copy=False)
nutrients = nutrients.rename(columns=col_mapping2, copy=False)
# print(frame4_1.keys())

# Merge tables:
ndata = pd.merge(nutrients, frame4_1, on='id', how='outer')
print(ndata.iloc[0])

# Plot a graph breaking down median amount of zinc in a member of each food group
result = ndata.groupby(['nutrient', 'fgroup'])['value'].quantile(0.5)
result['Zinc, Zn'].sort_values().plot(kind='barh')
# plt.show()

# Determine which food is the most dense in each nutrient
by_nutrient = ndata.groupby(['nutgroup', 'nutrient'])
# idxmax gets index of highest value, xs gets cross-section (all values for a subset)
get_maximum = lambda x: x.xs(x.value.idxmax())
max_foods = by_nutrient.apply(get_maximum)[['value', 'food']]
max_foods['food'] = max_foods['food'].str[:50]
print(max_foods.loc['Amino Acids']['food'])