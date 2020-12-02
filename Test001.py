'''
Simple Regression scatter plot. X axis is growth in the number of supercomputers and Y axis is GDP per capita growth.
Each point is a country
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# fatching and extrating needed data from csv file
data_sc = pd.read_csv("TOP500_202011.csv", usecols=["Country"])
data_sc = data_sc.values
data_y = []
for i in data_sc:
    data_y.append(i[0])
data_gdp = pd.read_csv("GDP.csv", usecols=["Economy","(millions of US dollars)"])
data_gdp = data_gdp.values
req_data = []
for i in range(len(data_gdp)):
    count = 0
    for j in data_y:
        if data_gdp[i][0] == j:
            count += 1

    if count == 0 :
        pass
    else:
        req_data.append([data_gdp[i][0],data_gdp[i][1],count])
data = pd.DataFrame(req_data)

print(data)

# dealing with outliers (replace with SD)
print(data[2].describe())
data.loc[data[2]>100,2]=int(data[2].std())

# loading the data
real_x = data.iloc[:, 2].values
real_x = real_x.reshape(-1, 1)
real_y = data.iloc[:, 1].values
real_y = real_y.reshape(-1, 1)

# applying simple regression
training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.1, random_state=0)
Lin_reg = LinearRegression()
Lin_reg.fit(training_x, training_y)
pred_y = Lin_reg.predict(testing_x)

# scatter plot with orignal data
plt.subplot(3,1,1)
plt.scatter(real_x, real_y, color='blue')
plt.title('Simple Regression Scatter Plot\n each dote represent a county')
plt.ylabel('GDP per captia Groth')

# scatter plot with trainind data
plt.subplot(3,1,2)
plt.scatter(training_x, training_y, color='blue')
plt.plot(training_x, Lin_reg.predict(training_x), color='red')
plt.ylabel('GDP per captia Groth')

# scatter plot with testing data
plt.subplot(3,1,3)
plt.scatter(testing_x, testing_y, color='blue')
plt.plot(training_x, Lin_reg.predict(training_x), color='red')
plt.xlabel('Training & Testing data\nnumbers of Supercomputer')
plt.ylabel('GDP per captia Groth')
plt.show()
print(pred_y,testing_y)