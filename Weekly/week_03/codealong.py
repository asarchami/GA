from __future__ import division

list_of_points =[(-1, 0), (1, 3), (2, 4), (3, 7), (4, 10)]

mean_x = sum(item[0] for item in list_of_points) / len(list_of_points)
mean_y = sum(item[1] for item in list_of_points) / len(list_of_points)

mean_xy = sum(item[0]*item[1] for item in list_of_points) / len(list_of_points)
mean_x2 = sum(item[0]**2 for item in list_of_points)

variance = mean_x2 - mean_x**2
covariance = mean_xy - (mean_x * mean_y)
beta = covariance / variance

alpha = mean_y - (beta * mean_x)

def predict(x, alpha, beta):
    return beta + alpha * x
print alpha
print beta

print predict(4, alpha, beta)
