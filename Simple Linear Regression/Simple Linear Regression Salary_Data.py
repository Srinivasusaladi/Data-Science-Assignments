"""

Simple Linear Regression Assignment
Salary_Data Dataset

"""

# Import the necessary libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# Import the  data
df=pd.read_csv("Salary_Data.csv")
list(df)

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.describe())
print(df.info())
df.isnull().sum()

# Distribution of features
sns.histplot(data=df, x='YearsExperience', kde=True, bins=10, color='skyblue', label='YearsExperience')
sns.histplot(data=df, x='Salary', kde=True, bins=10, color='orange', label='Salary')
plt.title('Distribution of YearsExperience and Salary')
plt.legend()
plt.show()

# Scatter plot between features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='YearsExperience', y='Salary', data=df, color='green')
plt.title('Scatter Plot between YearsExperience and Salary')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()

# Pairwise correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Box plots for each feature
sns.boxplot(data=df[['YearsExperience', 'Salary']])
plt.title('Box Plot for YearsExperience and Salary')
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

#Split X and Y
Y=df["Salary"]
X=df[["YearsExperience"]]

# Data partition
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

# Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,Y_train)

# predictions
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
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='red')
plt.title('Actual vs Predicted values on Test Set')
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.show()







