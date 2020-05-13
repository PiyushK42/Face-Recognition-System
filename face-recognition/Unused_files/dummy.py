import pandas as pd
import csv
names={}
def from_excel_to_csv():
    df = pd.read_excel('data.xlsx')
    df.to_csv('./data.csv')
def getdata():
    with open('data.csv','r') as f:
        data = csv.reader(f)
        next(data)
        lines = list(data)
        for line in lines:
            names[int(line[0])+1] = line[1]
    print(names)
from_excel_to_csv()
getdata()
