import pandas as pd
import matplotlib.pyplot as plt

years = range(1880, 2011)

pieces=[]
columns = ['name', 'sex', 'births']

for year in years:
    path = f'/home/ubuntu/Documents/PythonForDataAnalysis/datasets/babynames/yob{year}.txt'
    tempFrame = pd.read_csv(path, names=columns)

    tempFrame['year'] = year
    pieces.append(tempFrame)

names = pd.concat(pieces, ignore_index=True)

total_births = names.pivot_table('births', index='year', columns='sex', aggfunc=sum)
total_births.plot()
total_births_name = names.pivot_table('births', index='year', columns='name', aggfunc=sum)

subset = total_births_name[['John', 'Ben', 'Claire', 'Marilyn', 'Tiffany']]
subset.plot(subplots=True, figsize=(12,10), grid=False)

plt.show()