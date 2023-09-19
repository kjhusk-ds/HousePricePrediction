import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from scipy.stats import kurtosis
#%%
import statsmodels.api as sm
import pylab as py
#%%
#import data

train = pd.read_csv(r"C:\Users\Keith\Downloads\house-prices-advanced-regression-techniques\train.csv")

test = pd.read_csv(r"C:\Users\Keith\Downloads\house-prices-advanced-regression-techniques\test.csv")
#%%
#shape of data
print(train.shape)
print(test.shape)
#%%
# look at data types
print(train.info())
print(test.info())
'''
Here we observe that the dataset is made up of a number of different data types and that there
are a number of missing values 
'''
#%%
# let's get the numerical features and their number *may contain Ordinal as well as Continuous

numerical_features = [col for col in train.columns if train.dtypes[col] != "object"]

#The response variable and the Id are numerical so we remove these
numerical_features.remove('SalePrice')
numerical_features.remove('Id')

print(f'The Numerical Features Are: {numerical_features}')
print(f'The number of numerical features is: {len(numerical_features)}')
#%%
#Now let's get the categorical features
categorical_features = [col for col in train.columns if train.dtypes[col] == "object"]

print(f'The Categorical Features Are: {categorical_features}')
print(f'The number of categorical features is: {len(categorical_features)}')

#%%
'''
let's take a look at the shape of the response variable (SalePrice)
'''

sns.displot(data=train, x="SalePrice", kde =True)
plt.show()
#%%

'''
The plot shows that the response variable is not normally distributed.
The 'weight' of the data is in the left tail of the distribution
'''
skewness = skew(train['SalePrice'], axis=0, bias= True)
print(skewness)

'''
Skewness is a statistical term and it is a way to estimate or measure the shape of a distribution.  It is an important statistical methodology that is used to estimate the asymmetrical behavior rather than computing frequency distribution. Skewness can be two types:

Symmetrical: A distribution can be called symmetric if it appears the same from the left and right from the center point.
Asymmetrical: A distribution can be called asymmetric if it doesn’t appear the same from the left and right from the center point.
Distribution on the basis of skewness value:

Skewness = 0: Then normally distributed.
Skewness > 0: Then more weight in the left tail of the distribution.
Skewness < 0: Then more weight in the right tail of the distribution. 
https://www.geeksforgeeks.org/how-to-calculate-skewness-and-kurtosis-in-python/

This is confirmed by the skewness value of 1.88
'''
#%%
'''
Kurtosis:
It is also a statistical term and an important characteristic of frequency distribution.
It determines whether a distribution is heavy-tailed in respect of the normal distribution. 
It provides information about the shape of a frequency distribution.

kurtosis for normal distribution is equal to 3.
For a distribution having kurtosis < 3: It is called playkurtic.
For a distribution having kurtosis > 3, It is called leptokurtic 
and it signifies that it tries to produce more outliers rather than the normal distribution.

https://www.geeksforgeeks.org/how-to-calculate-skewness-and-kurtosis-in-python/
'''

kurt = kurtosis(train["SalePrice"],axis=0,bias=True)
#%%
print(kurt)
'''
The kurtosis value of 6.5 indicates the response variable is high also indicating distribution is
not normal, and we may find more outliers of SalePrice among the data. There is more weight in the tails meaning higher probablity of finding anomalous
results https://www.linkedin.com/pulse/deviations-from-normality-ie-normal-distribution-ammar-a-raja/

However, Some Scholars recommend a  +/- 3 rule of thumb for kurtosis cut-offs.
The values for asymmetry and kurtosis between -2 and +2 are considered acceptable in order to 
prove normal univariate distribution (George & Mallery, 2010). Hair et al. (2010)
and Bryne (2010) argued that data is considered to be normal if skewness is between ‐2 to +2
and kurtosis is between ‐7 to +7. More rules of thumb attributable to Kline (2011) are given here.
Curran et al. (1996) suggest these same moderate normality thresholds of 2.0 and 7.0 for skewness and kurtosis respectively when assessing multivariate normality
https://imaging.mrc-cbu.cam.ac.uk/statswiki/FAQ/Simon#:~:text=A%20value%20of%206%20or,is%20a%20very%20benign%20deviation.

So, the data may not be so far from normal to worry too much

But, For parametric (normally distributed, symmetrical) data, the mean and SD are the
appropriate measures of central tendency and variability of the data. 
For non-parametric data, the median is the appropriate central tendency measure and the IQR
is the appropriate measure of the variability of the data.
https://academic.oup.com/bjaed/article/7/4/127/466523
which is something to bear in mind
'''
#%%
sns.boxplot(data=train,x="SalePrice")
plt.show()
#shows that data is heavily skewed by many outliers
#%%
sm.qqplot(train["SalePrice"], line ='45')
py.show()
#%%
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('me')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/`