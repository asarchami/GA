import numpy as np
import scipy.stats as stats
import csv
# import seaborn as sns
# %matplotlib inline

import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
# plt.show()

csv_path = "sales_info.csv"

list_data = []
import csv

with open(csv_path, 'r') as f:
    for row in csv.reader(f):
        list_data.append(row)
# with open(sales_csv_path, 'r') as f:
#   sales_data = [x.split() for x in f]
print list_data
