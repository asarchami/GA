import csv
import os

with open('sales.csv', 'rU') as f:
    reader = csv.reader(f)
    for row in reader:
        print row

data = [['Me', 'You'],
       ['Hello', 'Goodbye'],
       ['Monday', 'Friday']]
with open('writeTest.csv', 'a') as fp:
    a = csv.writer(fp, delimiter=',')
    a.writerows(data)

with open('writeTest.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print row
