# Hypothesis_Testing_Assignment


"""
P1:
A F&B manager wants to determine whether there is any significant difference in
the diameter of the cutlet between two units. A randomly selected sample of cutlets 
was collected from both units and measured? Analyze the data and draw inferences at 5% 
significance level. Please state the assumptions and tests that you carried out to 
check validity of the assumptions

"""
# Two sample Z test

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Cutlets.csv")
list(df)

# EDA
print(df.head())
print(df.describe())
print(df.info())

# Distribution of features
sns.histplot(data=df, x='Unit A', kde=True, bins=20, color='skyblue', label='Unit A')
sns.histplot(data=df, x='Unit B', kde=True, bins=20, color='orange', label='Unit B')
plt.title('Distribution of Unit A and Unit B')
plt.legend()
plt.show()

# Scatter plot between Unit A and Unit B
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Unit A', y='Unit B', data=df, color='green')
plt.title('Scatter Plot between Unit A and Unit B')
plt.xlabel('Unit A')
plt.ylabel('Unit B')
plt.show()

# Pairwise correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Box plots for each feature
sns.boxplot(data=df[['Unit A', 'Unit B']])
plt.title('Box Plot for Unit A and Unit B')
plt.show()


df["Unit A"]
df["Unit B"]

# Zcalc,Pvalue
from scipy import stats
zcalc,pval = stats.ttest_ind(df["Unit A"],df["Unit B"])

zcalc
pval

# Compare p_value with (Significane Level)
alpha=0.05
if pval < alpha:
    print("Ho is rejected and H1 is accepted")
if pval >  alpha:
    print("H1 is rejected and Ho is accepted")

'''
Inferences:
    
H1 is rejected and Ho is accepted,means the null hypothesis is accepted,So there is not
enough evidence to conclude a significant difference in the diameter of cutlets between the two units.

'''





#==============================================================================

"""
P2:
A hospital wants to determine whether there is any difference in the average Turn 
Around Time (TAT) of reports of the laboratories on their preferred list. They collected 
a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample 
collected to report dispatch.
   
Analyze the data and determine whether there is any difference in average TAT among the 
different laboratories at 5% significance level.

"""
# ANOVA : Analysis of variance

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, f_oneway

# Load the dataset
df = pd.read_csv('LabTAT.csv')
list(df)

# EDA
print(df.head())
print(df.describe())
print(df.info())

# Pairwise correlation matrix (Heatmap)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Distribution of features (Histogram)
for column in df.columns:
    sns.histplot(df[column], kde=True, bins=20, label=column)
plt.title('Distribution of Laboratories')
plt.legend()
plt.show()

# Box plots for each features
sns.boxplot(data=df)
plt.title('Box Plot for features')
plt.show()

# Check normality for each group (Laboratory)
for column in df.columns:
    stat, p_value = shapiro(df[column])
    print(f"Shapiro-Wilk test for normality for {column}: p-value = {p_value}")

# Check homogeneity of variance
stat, p_value = levene(df['Laboratory 1'], df['Laboratory 2'], df['Laboratory 3'], df['Laboratory 4'])
print(f"homogeneity of variance: p-value = {p_value}")

# One-way ANOVA
stat, p_value = f_oneway(df['Laboratory 1'], df['Laboratory 2'], df['Laboratory 3'], df['Laboratory 4'])
print(f"One-way ANOVA: p-value = {p_value}")

# Compare p-value with significance
if p_value < 0.05:
    print("Ho is rejected and H1 is accepted")
else:
    print("H1 is rejected and Ho is accepted")

"""
Inferences:
Ho is rejected and H1 is accepted.
Reject the null hypothesis,means there is a significant difference in average TAT among the different laboratories.

"""





#=============================================================================

"""
P3:
Sales of products in four different regions is tabulated for males and females. 
Find if male-female buyer rations are similar across regions.

"""

# Chi - square test
# Chi - square test of independence:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Read the data from the CSV file
df = pd.read_csv('BuyerRatio.csv')
list(df)
df.head()

# EDA
print(df.head())
print(df.describe())
print(df.info())

# Bar plot for Buyer Ratio by Region
df_melted = pd.melt(df, id_vars=['Observed Values'], value_vars=['East', 'West', 'North', 'South'], var_name='Region', value_name='Buyer Ratio')
plt.figure(figsize=(10, 6))
sns.barplot(x='Region', y='Buyer Ratio', hue='Observed Values', data=df_melted, palette='viridis')
plt.title('Buyer Ratio by Region')
plt.xlabel('Region')
plt.ylabel('Buyer Ratio')
plt.show()


# Drop the 'Observed Values' column as it's not needed in the analysis
df = df.drop('Observed Values', axis=1)

# Perform the chi-square test
chi2, p, dof, expected = chi2_contingency(df)

# Print the results
print(f"Chi-Square Value: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(pd.DataFrame(expected, index=['Males', 'Females'], columns=df.columns))

# Compare p_value with (Significane Level)
if p < 0.05:
    print("Ho is rejected and H1 is accepted.")
else:
    print("H1 is rejected and Ho is accepted.")

"""
Infernces:
H1 is rejected and Ho is accepted.
Fail to reject the null hypothesis: Male-female buyer ratios are similar across regions.

"""






#=============================================================================

"""
P4:
TeleCall uses 4 centers around the globe to process customer order forms. They audit a 
certain %  of the customer order forms. Any error in order form renders it defective 
and has to be reworked before processing.  The manager wants to check whether the 
defective %  varies by centre. Please analyze the data at 5% significance level and help 
the manager draw appropriate inferences

"""

# Chi - square test
# Chi - square test of independence:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Load your dataset
df = pd.read_csv("Costomer+OrderForm.csv")
list(df)

# EDA
print(df.head())
print(df.describe())
print(df.info())

# Countplot for order status in each country
plt.figure(figsize=(10, 6))
sns.countplot(data=df.melt(), x='variable', hue='value', palette='Set2')
plt.title('Order Status in Each Country')
plt.xlabel('Country')
plt.ylabel('Count')
plt.show()

columns = ['Phillippines', 'Indonesia', 'Malta', 'India']

# Iterate through each pair of columns and perform chi-square test
for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        contingency_table = pd.crosstab(df[columns[i]], df[columns[j]])
        
        # Display the contingency table
        print(f"Contingency Table for {columns[i]} and {columns[j]}:")
        print(contingency_table)
        
        # Perform the chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Display test statistics
        print(f"Chi-Square: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of Freedom: {dof}")
        
        # Compare p_value with (Significane Level)
        alpha = 0.05
        if p < alpha:
            print("Reject the null hypothesis(Ho),means there is a evidence of significant difference in defective percentages.")
        else:
            print("Fail to reject the null hypothesis(Ho),means there is no significant difference in defective percentages.")




