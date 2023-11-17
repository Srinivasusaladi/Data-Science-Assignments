# Basic Statistics Level-2 Assignment Code

#------------------------------------------------------------------------------

# Set-1 - Descriptive Statistics and Probability

# Question No1
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
Measure_x=pd.Series([24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,
25.99,39.42,26.71,35.00]) 
#Boxplot
plt.figure(figsize=(8,4)) 
sns.boxplot(Measure_x) 
plt.show()

Measure_x.mean()
Measure_x.std()
Measure_x.var()


# Question No4
n = 5
p = 1/200
from scipy.stats import binom
Bi=binom(n=5,p=0.005)
#P(x<1)
P = 1-Bi.cdf(0)


#------------------------------------------------------------------------------

# Set-2 - Normal distribution, Functions of Random Variables

# Question No1
from scipy.stats import norm
# Normal distribution
N=norm(55,8)
# P(x>60)
P = 1-N.cdf(60)

# Question No2
# A)
from scipy.stats import norm
N=norm(38,6)
#P(x>44)
P = 1-N.cdf(44)

from scipy.stats import norm
N=norm(38,6)
#P(38-44)
P = N.cdf(44)-N.cdf(38)

# B)
from scipy.stats import norm
N=norm(38,6)
#P(x<30)
P = N.cdf(30)

# Question No4
from scipy.stats import norm
z_value = norm.ppf(0.005)

from scipy.stats import norm
z_value = norm.ppf(0.995)

a = (20 * (-2.57)) + 100 
b = (20* 2.57) + 100 



# Question No5
# A)
from scipy import stats
stats.norm.interval(0.95, loc = 540, scale = 225)

# C)
from scipy.stats import norm
N=norm(5,3)
#P(x<0)
P1 = N.cdf(0)

from scipy.stats import norm
N=norm(7,4)
#P(x<0)
P2 = N.cdf(0)


#------------------------------------------------------------------------------

# Set-3 - Confidence Intervals 

# Question No5
import numpy as np 
from scipy import stats 
Z=(0.046-0.05)/(np.sqrt((0.05*(1-0.05))/2000))
P_value=1-stats.norm.cdf(abs(Z))








































