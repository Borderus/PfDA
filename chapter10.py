# Import libraries

from pandas import Series, DataFrame
from datetime import datetime, timedelta, date

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

# *********************** Time Series *************************
    
print(f"Current datetime: {datetime.now()}")
print(f"Current year: {datetime.now().year}")
print(f"Current month: {datetime.now().month}")
print(f"Current day: {datetime.now().day}")
print()
def_dt = datetime(2023, 9, 21, 13, 4)
print(f"Self-defined datetime: {def_dt}")
print()
dt_delta = datetime.now() - def_dt
print(f"Delta between these: {dt_delta}")
print(f"Days between these: {dt_delta.days}")
print(f"Seconds between these: {dt_delta.seconds}")
print()
print(f"Twice this time ago: {datetime.now() - 2*dt_delta}")
print()
print(f"This day: {date.today()}")
print()
bday=datetime(1996, 5, 2, 13, 30)
print(f"Normal format: {bday}")
print(f"New format: {bday.strftime('%I:%M%p, %A %d %b %Y')}")
print()
datestring = "2024-01-03"
datestring_dt = datetime.strptime(datestring, "%Y-%m-%d")
print(f"Date String: {datestring}")
print(f"Read-in of Date String: {datestring_dt}")
print()

# Enough fooling with datetimes - onto pandas and datetimes

# Pandas has a to_datetime methpd to ocnvert datetimes:
datestrs = ["2011-07-06 12:00:00", "2011-08-06 00:00:00", np.nan]
datestrs_dts = pd.to_datetime(datestrs)
print(datestrs)
print(datestrs_dts)
print()

# ********** Time Series Basics **********

dates = [datetime(2011,i,1) for i in range(1,13)]
series1_1 = pd.Series(10*np.random.standard_normal(12), index=dates)
printSeries(1,1)

# Indexes automatically align - in the below, only alternate values are added on
series1_2 = series1_1 + series1_1[::2]
printSeries(1,2)

# The individual records in the index are Timestamp objects. These are effectively 
# more capable datetimes, with greater precision and more capabilities, e.g. timezones
picked_lab = series1_2.index[4]
print(picked_lab)
print(series1_2[picked_lab])
print()

# We can also pass a string that can be interpreted as the date
print(series1_2["2011-05-01"])
print()

# The pd.date_range function can allow us to generate a range
series1_3 = pd.Series(np.random.standard_normal(731), index=pd.date_range("2019-01-01", periods=731))
printSeries(1,3)

# We can select years or years and months with strings
series1_4 = series1_3['2020']
printSeries(1,4)
series1_5 = series1_3['2020-05']
printSeries(1,5)

# We can slice with datetime objects:
series1_6 = series1_3[datetime(2020, 6, 4):datetime(2020, 8, 10)]
printSeries(1,6)

# Duplicates can exist
dates = pd.DatetimeIndex(["2000-01-01", "2000-01-02", "2000-01-02", "2000-01-02", "2000-01-03"])
series1_7 = pd.Series(np.arange(5), index=dates)
printSeries(1,7)

# ********** Date Ranges, Frequencies and Shifting **********

# We can create date ranges with the pandas date_range method

print(pd.date_range(start='2023-12-01', periods=7))
print()
print(pd.date_range(end='2023-12-07', periods=7))
print()
# We can use different frequencies
print(pd.date_range(start='2023-12-01', periods=7, freq="W"))
print()
# We can also set times to midnight using the normalise option
print(pd.date_range(start='2023-12-01 12:56:34', periods=7, normalize=True))
print()
# We can import frequencies and start to use combinations
print(pd.date_range(start='2023-12-01', end ='2023-12-03', freq="4H"))
print()
print(pd.date_range(start='2023-12-01', end ='2023-12-03', freq="2H15MIN"))
print()
# Some frequencies describe points in time that aren't evenly spaced. For example,
# X months away. We call these anchored offsets.

# Another fun one is WOM - week of month. Let's use it to calculate book club
print(pd.date_range(start=datetime.today(), periods=12, freq="WOM-3WED"))
print()

# *** Shifting ***

# Shifting refers to shifting data backward and forward through time. Both Series
# and DataFrams have a simple function for this:

series2_1 = pd.Series(np.random.standard_normal(4), index=pd.date_range('2023-12-01', periods=4, freq='M'))
printSeries(2,1)
series2_2 = series2_1.shift(1)
printSeries(2,2)
# As we can see, each row gets budged along and missings are created
# This method can also be used to calculate percentage changes
frame2_3 = DataFrame(series2_1, columns=['value'])
frame2_3['pct_diff'] = (frame2_3['value']/frame2_3['value'].shift(1))-1
printFrame(2,3)
# Other frequencies can be passed, allowing some flexibility
series2_4 = series2_1.shift(2, freq='D')
printSeries(2,4)
# We can also use the offsets with datetimes:
from pandas.tseries.offsets import Day, MonthEnd
now = datetime.today()
print(now)
print()
print(now + 3*Day())
print()
# We can do this with anchored offsets - the first increment will 'roll forward'
# a date
print(now + MonthEnd())
print()
print(now + 2*MonthEnd())
print()
# There exist explicit rollforward and rollback methods for this too.
# This can also be used to group monthly data, though the resample method is much more common.

# ********** Time Zone Handling **********

# In pandas, Timezones are handled with the pytz library

import pytz
tz = pytz.timezone("America/Los_Angeles")
series3_1 = pd.Series(abs(np.random.standard_normal(6)), pd.date_range(start="2024-03-14 15:34:57", periods=6, freq="90T"))
printSeries(3,1)
# This is called a time zone naive index
# We could correct this by defining our timezone initially:
series3_2 = pd.Series(abs(np.random.standard_normal(6)), pd.date_range(start="2024-03-14 15:34:57", periods=6, freq="90T", tz='UTC'))
printSeries(3,2)
# Or we can convert through the tz_localize method:
series3_3 = series3_1.tz_localize("UTC")
printSeries(3,3)
# Once the dataframe has been localised, we can use tz_convert to change it
series3_4 = series3_3.tz_convert("America/New_York")
printSeries(3,4)
# tz_localize and tz_convert are also methods on a DatetimeIndex
# These also exist for timestamp objects

# Now, let's have fun with DST
series3_5 = pd.date_range(start="2024-03-30 21:00:00", periods=24, freq='15T', tz='Europe/London')
printSeries(3,5)

# Finally, time-zone naive data cannot be combined with time-zone aware data,
# and if two different timezones are used, we convert to UTC

# ********** Periods and Period Arithmetic **********

# Periods represent timespans and are represented by a Pandas class

p = pd.Period("2022", freq="A-DEC")
print("p =", p)

# The differences between periods can be represented with date offsets
p2 = pd.Period("2024", freq="A-DEC")
print("p2 =", p2)
print("p2 - p =", p2-p)

# Ranges of periods can be constructed with the period_range function
periods = pd.period_range("2020-01-01", "2021-03-06", freq="M")
print("periods = ",periods)
print()

# These can then be used as an index - the PeriodIndex class.
series3_6 = pd.Series(np.random.randn(15), periods)
printSeries(3,6)

# We can convert these with the .asfreq() method:
period_yrs = periods.asfreq(freq='Y')
series3_7 = pd.Series(np.random.randn(15), period_yrs)
printSeries(3,7)

# We can convert the entire time series with this
# Interestingly, the info is not kept that we lost by converting from M to Y
series3_8 = series3_7.asfreq(freq='M')
printSeries(3,8)

# We can also work with quarters
pq = pd.Period("2023Q1", freq="Q-APR")
print("pq =", pq)

# We could then, if we wanted, use this alongside the B (working day) frequency
# to get working days of the quarter

# We can also get quarterly period ranges:
periodq = pd.period_range("2011Q3", "2012Q4", freq="Q-APR")
periodqm = pd.period_range("2011Q3", "2012Q4", freq="Q-APR").asfreq(freq="M")
print(periodq)
print(periodqm)

# We can pass columns to the PeriodIndex constructor to create them too:
frame3_9 = pd.read_csv("/home/ubuntu/Documents/PythonForDataAnalysis/examples/spx.csv")
printFrame(3,9)

frame3_9['Date'] = pd.to_datetime(frame3_9['Date'], format='%Y-%m-%d %H:%M:%S').dt.to_period(freq='D')
frame3_9 = frame3_9.set_index('Date')
printFrame(3,9)
print(frame3_9.columns)
print(frame3_9.loc['1990-02-01'])

# ******* Resampling and Frequency Conversion *******

# Resampling refers to the process of converting a frequency
# Downsampling means to reduce the frequency
# Upsampling means to increase it
# These don't cover everything - changing W-WED to W-FRI is
# resampling but neither of the above.

# The workhorse in this field is the .resample() method, which
# is combined with an aggregation function to calculate (similar to groupby)

dates = pd.date_range("2020-01-01", periods=100)
series4_1 = pd.Series(np.random.standard_normal(100), index=dates)
printSeries(4,1)

series4_2 = series4_1.resample("M").mean()
printSeries(4,2)

# Resample Method Arguments

# Argument     Desc
# rule         String, DateOffset or TimeDelta indicating desired freq (e.g., 'M', '5min' or Second(5))
# axis         Axis to resample on
# fill_method  How to interpolate when upsampling; by default does nothing
# closed       In dowsampling, which end of the interval is closed
# label        In downsampling, how to label the result (the 'right' or 'left' ends of the interval)
# limit        Maximum number of periods to fill when forward or backfilling
# kind         Aggregate to Periods vs Timestamps
# convention   When upsampling periods, the end to use ('start' or 'end')
# origin       The 'base' timestamp to work from - see docs for more
# offset       An offset timedelta added to the origin - defaults to none

# *** Downsampling ***

# This joins bins (periods of time) into larger bins, which are fundamentally half-open intervals of time.
# There are a couple of things to think about when doing this:
# 
#     - Which end of the interval is closed
#     - Whether to lable the interval with the start or end values
# 
# Let's use an example:

dates = pd.date_range("2024-01-01", periods=12, freq="1min")
series4_3 = pd.Series(np.arange(12), index=dates)
printSeries(4,3)

# We can resample this in a few ways:
series4_4 = series4_3.resample('5min').sum()
printSeries(4,4)

series4_4 = series4_3.resample('5min', closed='right').sum()
printSeries(4,4)

series4_4 = series4_3.resample('5min', closed='right', label='right').sum()
printSeries(4,4)

# NOTE: There is no default for which end of an interval is closed.

# We can add an offset to the given values if we like:
from pandas.tseries.frequencies import to_offset
series4_5 = series4_4.copy()
series4_5.index = series4_5.index + to_offset('-1s')
printSeries(4,5)

# Open-high-low-close resampling

# In finance, a common way to want to aggregate is by showing the open value, high, low and close -
# we have an aggregate function that does this for us

series4_6 = pd.Series(np.random.permutation(np.arange(12)), index=dates)
printSeries(4,6)

series4_7 = series4_6.resample("5min", closed="left").ohlc()
printSeries(4,7)

# *** Upsampling ***

# Upsampling is needed to convert from a low frequency to a high one, where no aggregation is needed.
# Let's consider a dataframe with some weekly data:

frame4_8 = pd.DataFrame(np.random.standard_normal((2,4)), index=pd.date_range("2024-05-01", periods=2, freq="W-WED"), 
                     columns=["Colorado", "Texas", "New York", "Ohio"])
printFrame(4,8)
# To convert without aggregation, we can use the as_freq method:
frame4_9 = frame4_8.resample("D").asfreq()
printFrame(4,9)
# We can try and fill in the missing values:
frame4_9 = frame4_8.resample("D").ffill()
printFrame(4,9)
# But this is a bit much. We can impose a limit on how far we ffill:
frame4_9 = frame4_8.resample("D").ffill(limit=2)
printFrame(4,9)
# The new index need not coincide with the old one either:
frame4_10 = frame4_8.resample("W-THU").ffill()
printFrame(4,10)

# *** Resampling with Periods ***

frame4_11 = pd.DataFrame(np.random.standard_normal((24,4)),
                         index=pd.period_range("1-2000", "12-2001", freq="M"),
                         columns=['Colorado', 'Texas', 'New York', 'Ohio'])
printFrame(4,11)

frame4_12 = frame4_11.resample("A-DEC").mean()
printFrame(4,12)

# Upsampling can be more nuanced - we have to decide which end of the interval to put the data
frame4_13 = frame4_12.resample("Q-DEC", convention='start').mean()
printFrame(4,13)
frame4_13 = frame4_12.resample("Q-DEC", convention='end').mean()
printFrame(4,13)

# Periods are more strict than timestamps - up/down-samples NEED to be for sub/super-periods

# *** Grouped Time Resampling ***

# Resampling is basically just grouping.
# But what if we want to also group? Take the following:

times = pd.date_range("2017-05-20 00:00", freq="T", periods=15)
frame4_14 = pd.DataFrame({"time": times.repeat(3),
                          "key": np.tile(['a', 'b', 'c'], 15),
                           "value": np.arange(45)})
printFrame(4,14)

# To resample using keys, we cna introduce the Pandas Grouper object:
time_key = pd.Grouper(freq="5min")
frame4_15 = frame4_14.set_index("time").groupby(["key", time_key]).sum()
printFrame(4,15)

# One constraint of this is that the time must be the inde of the Series or DataFrame

# ************* Moving Window Functions *************

# The following methods are helpful for noisy data.
# Now, let's get some noisy data

frame5_1 = pd.read_csv('/home/ubuntu/Documents/PythonForDataAnalysis/examples/stock_px.csv', parse_dates=True, index_col=0).resample("B").ffill()
printFrame(5,1)

# Let's plot the Apple index alongside a 250 day rolling average
fig, axes = plt.subplots(3,2)

axes[0,0].plot(frame5_1["AAPL"])
axes[0,0].plot(frame5_1["AAPL"].rolling(250).mean())

# Rolling returns missings if it finds any missing values
# We can create a tolerance with the min_periods argument
axes[1,0].plot(frame5_1["AAPL"].pct_change().rolling(250, min_periods=10).std())
# We can use the expanding method to create an expanding aggregate (i.e. include all
# records before the current one)
axes[1,0].plot(frame5_1["AAPL"].pct_change().rolling(250, min_periods=10).std().expanding().mean())

# Calling a moving window function on a dataframe applies it to all columns
axes[0,1].plot(np.log10(frame5_1.rolling("60D").mean()))

# *** Exponentially Weighted Functions ***

# This allows us to give more recent records an exponential weighting, for say for a mean
frame5_2 = frame5_1['AAPL'].loc["2006":"2007"]
printFrame(5,2)

axes[1,1].plot(frame5_2)
axes[1,1].plot(frame5_2.rolling(30, min_periods=20).mean())
axes[1,1].plot(frame5_2.ewm(span=30).mean())

# *** Binary Moving Window Functions ***

# Some things, like correlation or covariance, operate on two series
# For example, lets see how well Apple's performance correlates with the market's

axes[2,0].plot(frame5_1["AAPL"].pct_change().rolling(125, min_periods=100).corr(frame5_1["SPX"].pct_change()))

# And we can do this for the lot of them:
axes[2,1].plot(frame5_1.pct_change().rolling(125, min_periods=100).corr(frame5_1["SPX"].pct_change()))
plt.show()

# Finally, we can define our own methods and apply them with the apply() method