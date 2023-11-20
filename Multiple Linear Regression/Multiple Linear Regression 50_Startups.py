"""
Multiple Linear Regression Assignment
50_Startups.csv Data file

"""

# Import the necessary libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the file
df = pd.read_csv("50_Startups.csv")
list(df)

# Exploratory Data Analysis (EDA) 
print(df.head())
print(df.describe())
print(df.info())
df.isnull().sum()

# split as numerical and catrgorical variables
categorical_columns = ["State"]
numerical_columns = ["R&D Spend","Administration","Marketing Spend","Profit"]

# Countplot for categorical featuers
for column in df[categorical_columns].columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f"Countplot for {column}")
    plt.show()

# Pairwise correlation matrix (Heatmap)
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Distribution of features (Histogram)
for column in df[numerical_columns].columns:
    sns.histplot(df[column], kde=True, bins=20, label=column)
plt.title('Distribution of Laboratories')
plt.legend()
plt.show()

# Box plots for each features
sns.boxplot(data=df)
plt.title('Box Plot for features')
plt.show()

# Identify outliers and impute with mean
outlier_threshold = 1.5
for column in df[numerical_columns].columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - outlier_threshold * iqr
    upper_bound = q3 + outlier_threshold * iqr
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Impute outliers with mean
    if not outliers.empty:
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = df[column].mean()

        # Optionally, print information about imputed values
        print(f"Imputed outliers for {column} with mean:")
        print(outliers)
        print("\n")
        
# Box plots for each features after the Outliers handling
sns.boxplot(data=df)
plt.title('Box Plot for features')
plt.show()   
                
# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
df[numerical_columns] = SS.fit_transform(df[numerical_columns])

# Label Encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for col in categorical_columns:
    df[col] = LE.fit_transform(df[col])

# Split the data into features (X) and target variable (Y)
X = df.drop('Profit', axis=1)
Y = df['Profit']

# Data partition
from sklearn.model_selection import train_test_split

# Model fitting LR
from sklearn.linear_model import LinearRegression
LR=LinearRegression()

# Metrics
from sklearn.metrics import mean_squared_error,r2_score

# Model Evaluation
# Validation set approach
training_errors = []
test_errors = []

for i in range(1,100):
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30,random_state=i)
    LR.fit(X_train,Y_train)
    # Predictions
    Y_pred_train = LR.predict(X_train)
    Y_pred_test  = LR.predict(X_test)
    training_errors.append(np.sqrt(mean_squared_error(Y_train ,Y_pred_train)))
    test_errors.append(np.sqrt(mean_squared_error(Y_test ,Y_pred_test)))
    
print("Average training error:",np.mean(training_errors).round(3))
print("Average test error:",np.mean(test_errors).round(3))

# Predictions (Entire Dataset)
LR.fit(X,Y)
Y_pred = LR.predict(X)

# Calculate R-squared on the entire dataset
r2 = r2_score(Y, Y_pred)
print("R square:", r2.round(3))

# Create a DataFrame to store R-squared values
r2 = pd.DataFrame({'Model': range(1, 100), 'R-squared': r2})
print(r2)

# Cube root transformation
Y_cbrt = np.cbrt(Y)
LR_cbrt = LinearRegression()
LR_cbrt.fit(X, Y_cbrt)
Y_pred_cbrt = LR_cbrt.predict(X)
Y_pred_inverse_cube = np.power(Y_pred_cbrt, 3)
# Calculate R-squared on the entire dataset
r2_cbrt = r2_score(Y, Y_pred_inverse_cube)
print("R square with cbrt transformation:", r2_cbrt)

# Visualizations
# A quantile-quantile (QQ) plot for entire dataset
residuals = Y - Y_pred
import statsmodels.api as sm
sm.qqplot(residuals, line='s')
plt.title('QQ Plot of Residuals')
plt.show()

# Scatter plot of actual vs predicted values on the test set
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred_test, color='blue')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], linestyle='--', color='red')
plt.title('Actual vs Predicted values on Test Set')
plt.xlabel('Actual Delivery Time')
plt.ylabel('Predicted Delivery Time')
plt.show()

# Plotting training and test errors (Learning curve)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 100), training_errors, label='Training Error', marker='o')
plt.plot(range(1, 100), test_errors, label='Test Error', marker='o')
plt.xlabel('Random State')
plt.ylabel('Root Mean Squared Error')
plt.title('Training and Test Errors Across Random Splits')
plt.legend()
plt.show()










