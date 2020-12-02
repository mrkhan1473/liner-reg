import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Salary_Data.csv')
real_x = data.iloc[:, 0].values
real_x = real_x.reshape(-1, 1)
real_y = data.iloc[:, 1].values
real_y = real_y.reshape(-1, 1)

plt.scatter(real_x, real_y, color='blue')
plt.title('Salary & Exp. plot')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()

training_x, testing_x, training_y, testing_y = train_test_split(real_x, real_y, test_size=0.3, random_state=0)
Lin_reg = LinearRegression()
Lin_reg.fit(training_x, training_y)
pred_y = Lin_reg.predict(testing_x)


plt.subplot(2,1,1)
plt.scatter(training_x, training_y, color='blue')
plt.plot(training_x, Lin_reg.predict(training_x), color='red')
plt.title('Salary & Exp. plot')
plt.xlabel('exp')
plt.ylabel('salary')

plt.subplot(2,1,2)
plt.scatter(testing_x, testing_y, color='blue')
plt.plot(training_x, Lin_reg.predict(training_x), color='red')
plt.xlabel('exp')
plt.ylabel('salary')
plt.show()