# Import libraries

from mpl_toolkits.basemap import Basemap
from pandas import Series, DataFrame
from numpy.random import randn
from datetime import datetime

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

# **************** Plotting and Visualisation *****************

# *** Test MatPlotLib with a quick plot ***
    
# plt.plot(np.arange(10))
# plt.show()
    
# *** Figures and Subplots ***
    
# # Plots in MatPlotLib reside within a Figure object. You can create this with plt.figure().
# # This will show an empty window: 
# fig = plt.figure()

# # We then have to add subplots to graph in:
# ax1 = fig.add_subplot(2,2,1)
# # This creates a 2x2 grid, and selects the first subplot (adding 1 subplot as it does)

# # Then, when we issue a command like plt.plot([1.5, 3.5, -2, 1.6]), matplotlib will draw 
# # on the most recently used subplot
# plt.plot([1.5, 3.5, -2, 1.6])

# # We can use an argument to this function to determine what sort of line we want:
# ax2 = fig.add_subplot(2,2,2)
# plt.plot(randn(50).cumsum(), 'k--')
# # The 'k--' argument will draw a black dashed line

# # The graphs can be drawn as methods, and we can specify what we want:
# ax4 = fig.add_subplot(2,2,4)
# ax4.scatter(np.arange(30), np.arange(30) + 3*randn(30))

# # n.b.1. There is a comprehensive list of plot types in the documentation

# # The subplots method is a convenience methods for quickly defining a figure and subplots
# #  - the subplots can be indexed which is particularly useful

# # We can adjust the spacing of subplots (from the edges or from each other) using the subplots_adjust method
# # left, bottom, top, right for edges - wspace and hspace for each other
# fig2, axes2 = plt.subplots(2, 2, sharex=True, sharey=True)
# for i in range(2):
#     for j in range(2):
#         axes2[i,j].hist(randn(500), bins=50, color='k', alpha=0.5)
# plt.subplots_adjust(wspace=0, hspace=0)

# plt.show() 

# *** Colors, Markers and Line Styles ***
    
# fig, axes = plt.subplots(3)
# series1_1 = randn(30).cumsum()

# # We can define the parts of the line like so:
# axes[0].plot(series1_1, color='#000000', linestyle='dashed', marker='o')
# # There is a shorthand for this: this instruction is equivalent to:
# axes[1].plot(series1_1, 'ko--')
# # For non-linear interpolation, try drawstyle='steps_post'
# axes[2].plot(series1_1, color='#00AAAA', linestyle='solid', marker='x', drawstyle='steps-post')

# plt.show()

# *** Ticks, Labels and Legends ***
    
# fig, axes = plt.subplots(2, sharex=True)
# axes[0].plot(randn(1000).cumsum())

# # Change the x axis ticks with set_xticks and set_xticklabels
# # Define the positions of the ticks:
# axes[0].set_xticks([0, 250, 500, 750, 1000])
# # Define the labels
# axes[0].set_xticklabels(['2019', '2020', '2021', '2022', '2023'], rotation=30, fontsize='small')
# axes[0].set_title("Illegal Immigration to Narnia, 2019-23")

# # Now to create multiple lines and a legend
# axes[1].plot(randn(1000).cumsum(), color='#000000', label='lions')
# axes[1].plot(randn(1000).cumsum(), color='#000000', linestyle='dashed', label='fauns')
# axes[1].plot(randn(1000).cumsum(), color='#000000', linestyle='dotted', label='beavers')
# axes[1].legend(loc='best')

# plt.subplots_adjust(hspace=0.3)
# plt.show()

# # *** Annotations and Drawing on a Subplot ***
    
# # Annotations and text can be added using the text, arrow and annotate functions
# # Let's plot the credit crunch

# fig1 = plt.figure()
# ax1 = fig1.add_subplot(1,1,1)

# # Read in example data
# frame1_1 = pd.read_csv('./examples/spx.csv', index_col=0, parse_dates=True)
# printFrame(1,1)

# # Draw line
# frame1_1.plot(ax=ax1, color='#007777')
# # Zoom in on area of interest
# ax1.set_xlim(['1/1/2007', '1/1/2011'])
# ax1.set_ylim([600, 1800])
# # Title
# ax1.set_title('Important Dates in 2008-2009 financial crisis')
# # Pick label locations
# crisis_data = [
#     (datetime(2007, 10, 11), 'Peak of Bull Market', 5),
#     (datetime(2008, 3, 12), 'Bear Stearns fails', 20),
#     (datetime(2008, 9, 15), 'Lehman bankruptcy', 50)
# ]
# # Create annotations for each of these
# for date, label, adjust in crisis_data:
#     ax1.annotate(label,
#                  xy=(date, frame1_1.loc[date.strftime("%Y-%m-%d")] + adjust),
#                  xytext=(date, frame1_1.loc[date.strftime("%Y-%m-%d")] + adjust + 150),
#                  arrowprops = {'facecolor': '#000000'},
#                  )

# # Now to make some shapes!
# # We define them with plt.<Shape>, and then add them with .add_patch()
# fig2, ax2 = plt.subplots()

# rect = plt.Rectangle((0.2, 0.75), 0.4, 0.15, color='#00FFFF', alpha=0.3)
# circ = plt.Circle((0.7, 0.2), 0.15, color='#0000FF', alpha=0.3)
# pgon = plt.Polygon([[0.15, 0.15], [0.35, 0.4], [0.2, 0.6]], color='#FF0000', alpha=0.5)
# ax2.add_patch(rect)
# ax2.add_patch(circ)
# ax2.add_patch(pgon)

# plt.show()

# # *** Saving Plots to File ***
    
# fig1.savefig('./figures/2008_Recession.jpg')

# # *** MatPlotLib Configuration ***
    
# # Altering defaults is done with the plt.rc() method

# # *************** Plotting Functions in Pandas ****************
    
# # *** Line Plots ***

# fig, ax = plt.subplots(2,5)


# # Series and DataFrame have a plot method for making these

# series2_1 = Series(randn(10).cumsum(), index=np.arange(0,100,10))
# series2_1.plot(ax=ax[0,0])

# # DataFrame method plots each column as a line on the samer subplot, 
# # creating a legend automatically

# frame2_2 = DataFrame(randn(10,4).cumsum(0), columns=['A', 'B', 'C', 'D'],
#                      index=np.arange(0,100,10))
# frame2_2.plot(ax=ax[1,0])

# # Series plot method  -  Description
# # label               -  Label for plot legend
# # ax                  -  Object to
# # style               -  Style string, like 'ko--'
# # alpha               -  Fill opacity
# # kind                -  Can be 'line', 'bar', 'barh', 'kde'
# # logy                -  Use log scaling on y axis
# # use_index           -  Use the index for tick labels
# # rot                 -  Rotation of tick labels
# # xticks              -  Value to use for x axis ticks
# # yticks              -  Value to use for y axis ticks
# # xlim                -  Value to use for x axis limits
# # ylim                -  Value to use for y axis limits
# # grid                -  Display grid

# # DataFrame plot method - Description
# # subplots              - Plot each column in a separate subplot
# # sharex                - If subplots=True, share the same x-axis
# # sharey                - If subplots=True, share the same y-axis
# # figsize               - Tuple to define size of figure
# # title                 - String defining title
# # legend                - Add a legend
# # sort_columns          - Plot columns in alphabetical order - by default uses existing order

# # *** Bar Plots ***

# # We can make bar plots by setting the kind argument to bar or barh
# series2_3 = Series(abs(randn(16)), index=list('abcdefghijklmnop'))
# series2_3.plot(kind='bar', ax=ax[0,1])
# series2_3.plot(kind='barh', ax=ax[1,1])

# # Grouped data can either be grouped or stacked
# frame2_4 = pd.read_csv('./examples/tips.csv')
# frame2_5 = pd.crosstab(frame2_4["day"], frame2_4["size"])
# # printFrame(2,4)
# # Let's remove parties of 1 or 6 as there aren't many
# frame2_5 = frame2_5[[2,3,4,5]]
# # printFrame(2,4)
# frame2_5.plot(ax=ax[0,2], kind='bar')
# frame2_5.plot(ax=ax[1,2], stacked=True, kind='bar', alpha=0.7)

# # *** Histograms and Density Plots ***

# # Histograms are basically bar charts counting how frequent values are

# frame2_6 = frame2_4.copy()
# frame2_6['tip_pct'] = frame2_6['tip'] / frame2_6['total_bill']
# frame2_6['tip_pct'].hist(bins=50, ax=ax[0,3])

# # We can produce a KDE (kernel density estimate) which can be thought of as a probability graph
# # This is done by randomly selecting normally distributed samples (kernels) and constructing
# # a graph out of these to smooth it out
# frame2_6['tip_pct'].plot(ax=ax[1,3], kind='kde')
# # And we can overlay it with a histogram
# frame2_6['tip_pct'].plot(ax=ax[0,4], kind='kde')
# frame2_6['tip_pct'].hist(bins=100, ax=ax[0,4], alpha=0.3)

# # *** Scatter Plots ***
# # Let's throw together a scatter plot using the macro data
# frame2_7 = pd.read_csv('./examples/macrodata.csv')
# # Select a set pof columns
# frame2_7 = frame2_7[['cpi', 'm1', 'tbilrate', 'unemp']]
# frame2_7 = np.log(frame2_7).diff().dropna()
# ax[1,4].scatter(frame2_7['m1'], frame2_7['unemp'])

# # *************** Exercise: Mapping Data from Haitian Earthquake ****************

# frame3_1 = pd.read_csv('./datasets/haiti/Haiti.csv')
# # Always start by looking at the table, the cols and doing a describe
# printFrame(3,1)
# print(frame3_1.columns)
# print(frame3_1.describe())
# # Let's remove the latitudes and longitudes that are far out and the missing incident categories
# frame3_1 = frame3_1[(frame3_1.LATITUDE > 18) & (frame3_1.LATITUDE < 20) &
#                     (frame3_1.LONGITUDE > -75) & (frame3_1.LONGITUDE < -70) &
#                     frame3_1.CATEGORY.notnull()]
# # print(frame3_1.describe())

# # Now we want to do a breakdown on the categories - but first we need to split them out

# # We start by creating a mapping from to codes to a description

# # Function to convert a comma-delimited string into a list
# def to_cat_list(catstr):
#   outlist = []
#   for x in catstr.split(','):
#      stripx = x.strip()
#      if stripx: # The 'if stringx' removes missing values
#         outlist.append(stripx)
#   return outlist

# # Function to get a set of all comma-delimited substrings within a series
# def get_all_categories(cat_series):
#   cat_sets = (set(to_cat_list(x)) for x in cat_series)
#   return sorted(set.union(*cat_sets))

# # Categories present as 4. French Name | English name
# # Create function to pick out English name
# def get_english(cat):
#    code, desc = cat.split('.')
#    if '|' in desc:
#       desc = desc.split('|')[1].strip()
#    return code, desc

# print(get_english('4. French Name | English name'))

# all_cats = get_all_categories(frame3_1.CATEGORY)
# # Generator for a dictionary
# english_mapping = dict(get_english(x) for x in all_cats)
# # print(english_mapping)

# # With this mapping created, we now need to select records by category
# # We'll make flag columns for this

# # Get a full list of mapping codes
# def get_code(seq):
#    return [x.split('.')[0] for x in seq if x]
# all_codes=get_code(all_cats)
# # print(all_codes)
# # Create an index with these:
# code_index = pd.Index(np.unique(all_codes))
# # Create an empty frame with this
# dummy_frame = DataFrame(np.zeros((len(frame3_1), len(code_index))))
# dummy_frame.index=frame3_1.index 
# dummy_frame.columns=code_index
# # Get category from each row of frame3_1 and mark it in dummy dataset
# for row in frame3_1.index:
#    cat = frame3_1.CATEGORY.loc[row]
#    codes = get_code(to_cat_list(cat))
#    dummy_frame.loc[row, codes] = 1
# # print(dummy_frame)
   
# # And attach this dummy frame full of flags to our data:
# frame3_2 = frame3_1.join(dummy_frame.add_prefix('category_'))
# printFrame(3,2)

# # Right - now let's make some graphs!
# # This bit will be mainly lifted-and-shifted
# def basic_haiti_map(ax=None, lllat=17.25, urlat=20.25, lllon=-75, urlon=-71):
# #    Create polar stereographoc Basemap instance
#   m= Basemap(ax=ax, projection='stere',
#              lon_0=(urlon+lllon)/2,
#              lat_0=(urlat+lllat)/2,
#              llcrnrlat=lllat, urcrnrlat=urlat,
#              llcrnrlon=lllon, urcrnrlon=urlon,
#              resolution='f'
#              )
#   m.drawcoastlines()
#   m.drawstates()
#   m.drawcountries()
#   return m

# fig, axes = plt.subplots(2,2, figsize=(12,10))
# fig.subplots_adjust(hspace=0.1, wspace=0.05)

# to_plot = ['2a', '1', '3c', '7a']

# shapefile_path = './datasets/haiti/PortAuPrince_Roads/PortAuPrince_Roads'

# for code, ax in zip(to_plot, axes.flat):
#   m = basic_haiti_map(ax)
#   if code == '7a':
#      m.readshapefile(shapefile_path, 'roads')
#   cat_data = frame3_2[frame3_2[f'category_{code}'] == 1]
#   x, y = m(cat_data.LONGITUDE, cat_data.LATITUDE)
#   m.plot(x, y, 'k.', alpha=0.5)
#   ax.set_title(f'{code}: {english_mapping[code]}')

# plt.show()

# fig.savefig('./figures/Haiti_Impact.jpg')