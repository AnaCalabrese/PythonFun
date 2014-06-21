import pandas as pd
import numpy as np

COLUMN_SEPARATOR = ','
housing_data = pd.DataFrame.from_csv('housing.csv', sep=COLUMN_SEPARATOR, header=None)
#print(housing_data)

AREA_INDEX = 4
SELLING_PRICE_INDEX = 13
x = housing_data[AREA_INDEX]
y = housing_data[SELLING_PRICE_INDEX]

a = []
b = []
for i in range(1, len(x)):
    a.append(x[i])
    b.append(y[i])

# regression = np.polyfit(x, y, 1)
regression = np.polyfit(a, b, 1)
print(regression[0])
print(regression[1])

# We need to generate actual values for the regression line.
r_x = []
r_y = []
for i in range(15):
    r_x.append(i)
    r_y.append(i*regression[0] + regression[1])

from bokeh.plotting import *

# fancy way of doing the same thing...
#r_x, r_y = zip(*((i, i*regression[0] + regression[1]) for i in range(15)))

output_file("regression.html", title="toy housing data")
line(r_x, r_y, color="magenta", line_width=2,
     title="Toy housing data", legend="linear regression")

# Specify that the two graphs should be on the same plot.
hold(True)
scatter(a, b, size = 12, marker="circle", color="black", alpha=0.5)
xaxis().axis_label = 'Area (x 1000 sq feet)'
yaxis().axis_label = 'Selling price '
show()
