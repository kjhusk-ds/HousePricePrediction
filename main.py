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

#%%
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('me')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/`