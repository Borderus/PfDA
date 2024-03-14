# Import libraries

from pandas import Series, DataFrame
from pandas_datareader import data
from bs4 import BeautifulSoup
from lxml import etree

import pandas as pd
import numpy as np
import csv
import json
import urllib3
import sqlite3
import requests
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
# ********** Data loading, storage and file formats ***********
# *************************************************************

# Reading and writing data in text format

# BASIC TABULAR READIN FUNCTIONS:
# read_csv: Load delimited data from a file, URL or file-like object. Use comma as default delimiter
# read_table: Load delimited data from a file, URL or file-like object. Use tab ('\t') as default delimiter
# read_fwf: Read data from a fixed-width column format (i.e. no delimiters)
# read_clipboard: Versions of read_table that pulls from the clipboard
    
frame1_1 = pd.read_csv("examples/ex1.csv")
printFrame(1,1)
# We can achieve the same results using read_table and specifying the delimiter:
frame1_1 = pd.read_table("examples/ex1.csv", sep=',')
printFrame(1,1)
# But, there won't always be a header row:
frame1_2 = pd.read_csv("examples/ex2.csv", header=None)
printFrame(1,2)
# But we can specify this:
names1_2 = ['a', 'b', 'c', 'd', 'message']
frame1_2 = pd.read_csv("examples/ex2.csv", names=names1_2)
printFrame(1,2)
# If we want to use the message as an index, we can also specify this
frame1_3 = pd.read_csv("examples/ex2.csv", names=names1_2, index_col="message")
printFrame(1,3)
# We can specify MultiIndexing by passing a list
frame1_4 = pd.read_csv("examples/ex2_5.csv", index_col=['key1', 'key2'])
printFrame(1,4)
# A table may not have a fixed delimiter - in this instance we'll pass it some regex
frame1_5 = pd.read_table("examples/ex3.txt", sep="\s+", index_col=0)
printFrame(1,5)
# We can skip rows we don't want
frame1_6 = pd.read_csv("examples/ex4.csv")
printFrame(1,6)
frame1_6 = pd.read_csv("examples/ex4.csv", skiprows=[0,2,3])
printFrame(1,6)
# Now onto dealing with missing values. Missing data is usually either not present or marked by
# a sentinel values (e.g. NA, -1, IND, NULL). We can also create them for bad data
frame1_7 = pd.read_csv("examples/ex5.csv")
printFrame(1,7)
frame1_7 = pd.read_csv("examples/ex5.csv", na_values=[1])
printFrame(1,7)
frame1_7 = pd.read_csv("examples/ex5.csv", na_values={"message":["one", "foo"], "something": ["two"]})
printFrame(1,7)

# READ_CSV/READ_TABLE FUNCTION ARGUMENTS

# path: String indicating target
# sep: Separator
# header: Row number to use for col names. Defaults to 0, set to None if no header row
# index_col: Specify an index column. Can be a single name of list (for MultiIndexing)
# names: List of column names, combine with header=None
# skiprows: Number of rows to skip (starting from 0) or list of rows to skip
# na_values: Sequence of values to replace with NA
# comment: Identify characters indicating start of comment to ignore
# parse_dates: Attempt to parse data to datetime. False by default, setting to True will attempt all 
#              columns. Can pass a list of columns to attempt. If element of list is tuple or list,
#              will attempt to combine to create a datetime.
# keep_date_col: If joining columns to parse date, don't drop joined columns. Default True
# converters: Dict containing column number of name mapping to functions. For example, {'foo': f}
#             would apply f to all values in foo
# dayfirst: When parsing ambiguous dates, specify dd/mm/yyyy format. Default false.
# date_parser: Function used to parse dates
# nrows: Number of rows to read in
# iterator: Return a TextParser object for reading file piecemeal
# chunksize: For iteration, set size of chunks
# skip_footer: Number of lines to ignore at end of file
# verbose: Add info
# encoding: Specify text encoding
# squeeze: If the data only contains one col return a series
# thousands: Specify whether to use '.' or ',' for thousands 

# Reading text files in pieces

frame1_8 = pd.read_csv("examples/ex6.csv")
printFrame(1,8)
print(frame1_8.describe())
print()
# We can specify a small number of rows to read with nrows
frame1_8 = pd.read_csv("examples/ex6.csv", nrows=5)
printFrame(1,8)
# To read it piecemeal, specify a chunksize as a number of rows:
chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)
# This creates a TextParser object, an Iterator that allows us to iterate over parts of the file
# For example, we cna iterate over ex6.csv, aggregating the counts in the key column like so:
chunker = pd.read_csv("examples/ex6.csv", chunksize=1000)

tot=Series([])
for piece in chunker:
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

tot = tot.sort_values(ascending=False)
print("Iterated sum of ex6.csv", "\n", tot, "\n", sep="")

# Writing data out to text format

# To view these, use cat in the terminal
# We can also write out:
frame1_7.to_csv("examples/out1.csv")
# We can use other delimiters (here writing to sys.stdout so it prints)
frame1_7.to_csv("examples/out2.csv", sep='|')
# Alter the sentinel values for missing values
frame1_7.to_csv("examples/out3.csv", na_rep='NULL')
# We can remove the row and column labels:
frame1_7.to_csv("examples/out4.csv", index=False, header=False)
# Or just keep certain columns and determine an order to write them in:
frame1_7.to_csv("examples/out5.csv", columns=['a', 'b', 'c'])
# Series can also work like this - they use to to_csv and from_csv methods.

# Manually working with defined formats

# We may receive dirty data that prevents the aforementioned methods from working.
# We can use the csv module for this.
f = open('examples/ex7.csv')
reader=csv.reader(f)
for line in reader:
    print(line)

# We can read this in more evenly with the zip function
# First get the lines:
lines = list(csv.reader(open('examples/ex7.csv')))
# Define the first line as the header, the rest as the content
header, values = lines[0], lines[1:]
# The zip function can then match these into even n-tuples and 
# we can pass these into a dictionary through list comprehension
data_dict = {h: v for h,v in zip(header, zip(*values))}
# And print
print(data_dict)
print(DataFrame(data_dict))

# We can make things easy for ourselves by defining CSV dialexts:
class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ' '
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL
# Though it can still take some munging.
reader = list(csv.reader(open("examples/ex3.txt"), dialect=my_dialect))
# Sort spaces in header first
newreader=[]
for value in reader[0]:
    if value != '':
        newreader.append(value)
header = ['index']+newreader
# Now spaces in values
values = reader[1:]
newvalues = []
for row in values:
    temprow = []
    for item in row:
      if item != '':
          temprow.append(item)
    newvalues.append(temprow)
# Then combine
data_dict = {h: v for h,v in zip(header, zip(*newvalues))}
print(DataFrame(data_dict))

# CSV Dialect Options:

# delimiter: Delimiter
# lineterminator: Line Terminator
# quotechar: Quote character for specials
# quoting: Set quoting policy (QUOTE_ALL, QUOTE_MINIMAL, QUOTE_NONNUMERIC, QUOTE_NON)
# skipinitialspace: Ignore whitespace after delimiter. Default wrongly False
# doublequote: How to handle quoting character inside a field
# escapechar: Escape character

# We can also write out manually

with open('examples/out6.csv', 'w+') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))

# JSON Data
    
# Time to play with some JSON files!
    
obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""
# We can convert this to a Python dictionary with json.loads
result = json.loads(obj)
print(result)
# We can take it back tp JSON with json.dumps
obj = json.dumps(result)
print(obj)
print()
# We can convert this to a dataframe, and can pass a list of JSON objects to the 
# DataFrame constructor
frame1_9 = DataFrame(result['siblings'], columns=['name', 'age'])
printFrame(1,9)

# XML and HTML - Web Scraping

# There are many Python libraries for this work
# We are going to use lxml due to good performance with large files

http = urllib3.PoolManager()
url = 'https://www.bbc.co.uk/news'
response = http.request('GET', url)
print(response.status)

# BEAUTIFUL SOUP
print("BEAUTIFUL SOUP READ-IN OF BBC NEWS:")
if response.status == 200:
    soup = BeautifulSoup(response.data, 'html.parser')

    links = soup.find_all('a')
    for link in links:
        print(link.get('href'))
response.release_conn()

# XML
print("LXML READ-IN OF BOOKS.XML")
path = 'examples/books.xml'
root = etree.parse(path).getroot()
# We can print the lot with a .tostring()
# print(etree.tostring(root))
# Or let's try to parse it into a DataFrame:
xmldata = []
for child in root:
  nodeDict = {}
  for node in child:
    nodeDict[node.tag] = node.text
  xmldata.append(nodeDict)

frame1_10 = DataFrame(xmldata)
printFrame(1,10)

# *************************************************************

# Binary Data Formats

# Pickle Format

# We can save frames down in the binary pickle format
frame2_1 = pd.read_csv("examples/ex1.csv")
printFrame(2,1)
frame2_1.to_pickle("examples/frame_pickle")
frame2_2 = pd.read_pickle("examples/frame_pickle")
printFrame(2,2)

# We can also use the HDF5 format

# This is a scientific format that creates an internal file-system-like structure
# Very good for large datasets if good at compression.
# Python has two interfaces for this, h5py and PyTables. Pandas uses PyTables for this.

store = pd.HDFStore("examples/mydata.h5")
store['obj1'] = frame2_2
store['obj1_col'] = frame2_2['a']
print(store)
print(store['obj1'])
store.close()

# We can even read in Excel files

frame2_3 = pd.read_excel("examples/ex1.xlsx", engine="openpyxl")
printFrame(2,3)

# *************************************************************

# Interacting with HTML and Web APIs

# PULLING SNOOKER SCORES WITH BEAUTIFUL SOUP

url = 'https://snookerscores.net/tournament-manager/2024-english-6-red-snooker-championship/results'
response=requests.get(url)
if response.status_code == 200:
  soup = BeautifulSoup(response.content, 'html.parser')

  rounds = soup.find_all(href=re.compile("https://snookerscores.net/scoreboard/match/.*"))
  roundList = []
  for round in rounds:
      roundList.append(round.text.strip())
      
  players = soup.find_all(href=re.compile("https://snookerscores.net/player/.*"))
  playerList = []
  for player in players:
      playerList.append(player.text.strip())

  outcomes = soup.find_all("div", class_=re.compile("col-2 text-nowrap.*"))
  scoreList = []
  for outcome in outcomes:
    outcome_str = outcome.get_text()
    scoreList.append(outcome_str)

  matchList = []

  if len(playerList) == 2*len(scoreList):
    for counter in range(len(scoreList)):
       matchDict = {"round" : roundList[counter],
                    "player1" : playerList[2*counter],
                    "p1score" : scoreList[counter][0],
                    "p2score" : scoreList[counter][2],
                    "player2" : playerList[2*counter+1]}
       matchList.append(matchDict)
  
  frame2_4 = DataFrame(matchList)
  printFrame(2,4)

#   Let's looks at the tournament progression of the winner:
  frame2_5 = frame2_4[np.logical_or(frame2_4["player1"] == "Daniel Womersley", frame2_4["player2"] == "Daniel Womersley")]
  print("Dan Womersley Tounrament Progress:")
  print(frame2_5)
  print()

# *************************************************************

# Interacting with Databases

# SQLITE
print("PLAYING WITH SQL:")
query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
c REAL, d INTEGER
);"""
 
con=sqlite3.connect(':memory:')
con.execute(query)
con.commit()

data=[('Atlanta', 'Georgia', 1.25, 6),
      ('Tallahassee', 'Florida', 2.6, 3),
      ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit

cursor = con.execute('select * from test')
rows=cursor.fetchall()
print(rows)
# We can pass these to the DataFrame constructor but best to add in the names
col_list = []
for item in cursor.description:
    col_list.append(item[0])
frame3_1 = DataFrame(rows, columns=col_list)
printFrame(3,1)
# Or just:
frame3_1 = pd.io.sql.read_sql('select * from test', con)
printFrame(3,1)

# MongoDB

# N.B. Section on MongoDB skipped due to incompatibilities causing connections to time out