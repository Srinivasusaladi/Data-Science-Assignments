"""
Multiple Linear Regression Assignment
ToyotaCorolla.csv Data file

"""

# Import the necessary libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read the data
df = pd.read_csv("ToyotaCorolla.csv", encoding='latin1')
list(df)

# Exploratory Data Analysis (EDA)
print(df.head())
print(df.describe())
print(df.info())
df.isnull().sum()

# Separate numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int', 'float']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Countplot for categorical featuers
for column in df[categorical_columns].columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=column, data=df)
    plt.title(f"Countplot for {column}")
    plt.show()

# Pairwise correlation matrix (Heatmap)
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(12, 8))
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
plt.figure(figsize=(20, 20))
sns.boxplot(data=df[numerical_columns])
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

# Visualize box plots after Handling outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(9, 4, i)
    sns.boxplot(x=col, data=df)
    plt.title(f'Box Plot of {col} (After Outlier Handling)')
plt.tight_layout()
plt.show()


# Split as X and Y variable
Y = df[["Price"]]
X = df[["Age_08_04", "KM", "HP", "cc", "Doors", "Gears", "Quarterly_Tax", "Weight"]]

# Standardization
SS = StandardScaler()
SS_X = SS.fit_transform(X)
SS_Y = SS.fit_transform(Y)  # Scaling the target variable

# Data partition
training_errors = []
test_errors = []

# Model fitting and evaluation using cross-validation
for i in range(1, 100):
    X_train, X_test, Y_train, Y_test = train_test_split(SS_X, SS_Y, test_size=0.30, random_state=i)

    # Model fitting
    LR = LinearRegression()
    LR.fit(X_train, Y_train)

    # Predictions
    Y_pred_train = LR.predict(X_train)
    Y_pred_test = LR.predict(X_test)

    # Calculate errors
    training_errors.append(np.sqrt(mean_squared_error(Y_train, Y_pred_train)))
    test_errors.append(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))

# Calculate metrics
print("Average training error (RMSE):", np.mean(training_errors).round(3))
print("Average test error (RMSE):", np.mean(test_errors).round(3))

# Predictions (Entire Dataset)
LR.fit(SS_X, SS_Y)
Y_pred = LR.predict(SS_X)

# Calculate R-squared on the entire dataset
r2 = r2_score(SS_Y, Y_pred)
print("R square:", r2.round(3))
# Create a DataFrame to store R-squared values
r2 = pd.DataFrame({'Model': range(1, 100), 'R-squared': r2})
print(r2)


# Visualizations
# A quantile-quantile (QQ) plot for entire dataset
residuals = SS_Y - Y_pred
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
