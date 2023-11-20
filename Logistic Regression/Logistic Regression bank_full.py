"""
Logistic Regression Assignment
Bank-full.csv file

"""

# Import the necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df=pd.read_csv("bank-full.csv",sep=';')
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
    plt.figure(figsize=(10, 8))
    sns.histplot(df[column], kde=True, bins=20, label=column)
plt.title('Distribution of Laboratories')
plt.legend()
plt.show()

# Box plots for each features
plt.figure(figsize=(12, 10))
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
plt.figure(figsize=(12, 10))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(9, 4, i)
    sns.boxplot(x=col, data=df)
    plt.title(f'Box Plot of {col} (After Outlier Handling)')
plt.tight_layout()
plt.show()

# Data Transformation
# Label Encoding
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
for col in categorical_columns:
    df[col] = LE.fit_transform(df[col])
df[categorical_columns]

# Standardization
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
SS_X = SS.fit_transform(df[numerical_columns])
pd.DataFrame(SS_X)

# Split as X and Y variable
Y = df["y"]
X = df.iloc[:,0:16]

# Data partion
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import accuracy_score

# Model fitting
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

# Cross validation
# validation set approach
training_accuracy=[]
test_accuracy=[]

for i in range(1,100):
    x_train,x_test,y_train,y_test=train_test_split(SS_X,Y,test_size=0.30,random_state=i)
    logreg.fit(x_train,y_train)
    # Predictions
    y_pred_train= logreg.predict(x_train)
    y_pred_test= logreg.predict(x_test)
    training_accuracy.append(accuracy_score(y_train,y_pred_train))
    test_accuracy.append(accuracy_score(y_test,y_pred_test))
    
print("Average training_accuracy :",np.mean(training_accuracy).round(3))
print("Average test_accuracy:",np.mean(test_accuracy).round(3))
print("Variance",np.mean(training_accuracy).round(3)-np.mean(test_accuracy).round(3))

from sklearn.metrics import classification_report
print(classification_report(y_train,y_pred_train))

# Confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Exact probabilities
logreg.predict_proba(SS_X)
Y_proba = logreg.predict_proba(SS_X)[:,1:]

# Visualizations
# ROC CURVE
from sklearn.metrics import roc_curve,roc_auc_score
fpr,tpr,dummy = roc_curve(Y,Y_proba)

import matplotlib.pyplot as plt
plt.scatter(x = fpr,y=tpr)
plt.plot(fpr,tpr,color='red')
plt.ylabel("True positive Rate")
plt.xlabel("False positive Rate")
plt.show()

print("AUC score:", roc_auc_score(Y,Y_proba))


# Variation in training and test accuracies
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), training_accuracy, label='Training Accuracy')
plt.plot(range(1, 100), test_accuracy, label='Test Accuracy')
plt.title('Training and Test Accuracies Across Different Random States')
plt.xlabel('Random State')
plt.ylabel('Accuracy')
plt.legend()
plt.show()






