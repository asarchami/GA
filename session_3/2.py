import numpy as np
import scipy.stats as sp
import matplotlib


chris_numbers = [1, 6, 10, 100, 1000]

print np.mean(chris_numbers)
print np.median(chris_numbers)
print sp.mode(chris_numbers)
print np.var(chris_numbers)
i = np.std(chris_numbers)
print i
print np.ptp(chris_numbers)
print sp.iqr(chris_numbers)


easy_graph = [25, 50, 75]
print np.var(easy_graph)
