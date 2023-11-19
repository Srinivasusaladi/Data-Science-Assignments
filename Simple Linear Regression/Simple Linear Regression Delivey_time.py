"""

Simple Linear Regression Assignment
Delivey_time Dataset

"""

# Import the necessary libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Import the  data
df=pd.read_csv("delivery_time.csv")
list(df)

#EDA
print(df.head())
print(df.describe())
print(df.info())
df.isnull().sum()

# Distribution of features
sns.histplot(data=df, x='Delivery Time', kde=True, bins=10, color='skyblue', label='Delivery Time')
sns.histplot(data=df, x='Sorting Time', kde=True, bins=10, color='orange', label='Sorting Time')
plt.title('Distribution of Delivery Time and Sorting Time')
plt.legend()
plt.show()

# Scatter plot between features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Delivery Time', y='Sorting Time', data=df, color='green')
plt.title('Scatter Plot between Delivery Time and Sorting Time')
plt.xlabel('Delivery Time')
plt.ylabel('Sorting Time')
plt.show()

# Pairwise correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Box plots for each feature
sns.boxplot(data=df[['Delivery Time', 'Sorting Time']])
plt.title('Box Plot for Delivery Time and Sorting Time')
plt.show()

# Identify and print rows with outliers for each column
outlier_threshold = 1.5
for column in df.columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    if not outliers.empty:
        print(f"Outliers in {column}:")
        print(outliers)
        print("\n")
# No "Outliers"

#Split as X(input variable) and Y(target variable)
Y=df["Delivery Time"]
X=df[["Sorting Time"]]

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# Predictions
Y_pred_train = LR.predict(X_train)
Y_pred_test  = LR.predict(X_test)

# Metrics
from sklearn.metrics import mean_squared_error,r2_score
error1 = np.sqrt(mean_squared_error(Y_train ,Y_pred_train))
print("Train Error [RMSE]:", error1.round(3))
print("R square:", r2_score(Y_train ,Y_pred_train).round(3))

error2 = np.sqrt(mean_squared_error(Y_test ,Y_pred_test))
print("Test Error [RMSE]:", error2.round(3))
print("R square:", r2_score(Y_test ,Y_pred_test).round(3))

# Calculate R-squared on the entire dataset
LR.fit(X, Y)
Y_pred = LR.predict(X)
r2 = r2_score(Y, Y_pred)
print("R square:", r2.round(3))
# Create a DataFrame to store R-squared values
r2 = pd.DataFrame({'Model':['Linear Regression'] , 'R-squared': r2})
print(r2)

# Scatter plot of actual vs predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred_test, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='red', linewidth=2)
plt.title('Actual vs Predicted values on Test Set')
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.show()


















