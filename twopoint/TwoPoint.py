import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.linear_model import *


"""
Trying to deduce the formula to compute the training time in Two Point's Hospital
"""


df = pd.read_csv('training_times.csv')

base_min_speed = df['min_time'].max()
base_max_speed = df['max_time'].max()

df['min_speed'] = base_min_speed / df['min_time']
df['max_speed'] = base_max_speed / df['max_time']

df['bonus'] = df['count'] * df['percent']

# To show the raw data
# plot.scatter(df['bonus'], df['min_time'], c='b', marker='.')
# plot.scatter(df['bonus'], df['max_time'], c='r', marker='.')
# plot.show()

# A linear model to show the progression
model = LinearRegression()
model.fit([[x] for x in df['count']], df['min_speed'].tolist())
print(model.coef_)

xs = np.linspace(0, 1500, num=1500)
ys = model.predict(xs.reshape(-1, 1))

plot.scatter(xs, np.ceil(base_max_speed / ys), c='b', marker='.')
plot.show()
