# Basic statistics level 1 Assignment

# Question No7
import pandas as pd 
Data=pd.read_csv('Q7.csv') 
X = Data[["Points", "Score", "Weigh"]]
X.describe()
X.mean()
X.median()
X.mode()
X.var()
X.std()



# Question No8
import numpy as np 
weights_of_patients = [108, 110, 123, 134, 135, 145, 167, 187, 199]
np.mean(weights_of_patients) 



# Question No9
# a)Q9_a.csv file
Data = pd.read_csv('Q9_a.csv') 
Data .skew() 
Data .kurtosis()

# b)Q9_b.csv file
Data = pd.read_csv('Q9_b.csv') 
Data.skew() 
Data.kurtosis()



# Question No11
import numpy as np 
from scipy import stats 
stats.norm.interval(alpha=0.94, loc=200, scale= 30/np.sqrt(2000)) # 94% CI 
stats.norm.interval(alpha=0.96, loc=200, scale= 30/np.sqrt(2000)) # 96% CI 
stats.norm.interval(alpha=0.98, loc=200, scale= 30/np.sqrt(2000)) # 98% CI



# Question No12
import numpy as np 
x=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56] 
np.mean(x) 
np.median(x) 
np.var(x) 
np.std(x) 



# Question No20
import pandas as pd 
from scipy.stats import norm 
Data=pd.read_csv("Cars.csv") 
Mean=Data['MPG'].mean() 
std=Data['MPG'].std() 

N=norm(Mean,std)
#P(MPG<40)
N.cdf(40)
#P(MPG>38)
1-N.cdf(38)
#P(20<MPG<50)
N.cdf(50)-N.cdf(20)



# Question No21
# a)Cars.csv file
import pandas as pd 
Data=pd.read_csv("Cars.csv") 
Data['MPG'].hist()
Data['MPG'].skew()

# a)wc-at.csv file
import pandas as pd 
Data=pd.read_csv("wc-at.csv") 
Data['AT'].hist()
Data['AT'].skew()
Data['Waist'].hist()
Data['Waist'].skew()



# Question No22
from scipy import stats 
stats.norm.ppf(0.95) 
stats.norm.ppf(0.97) 
stats.norm.ppf(0.80) 



# Question No23
from scipy import stats 
stats.t.ppf(0.975,df=24) 
stats.t.ppf(0.98,df=24) 
stats.t.ppf(0.995,df=24) 



# Question No24
import numpy as np
import scipy.stats as stats
t_score = (260-270)/(90/np.sqrt(18))
t_score = -0.471
df = 17
stats.t.cdf(t_score, df = 17)



