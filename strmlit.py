# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False) 

st.title('House Price Prediction')

# %%
train_clean = pd.read_csv('train.csv')
test_clean = pd.read_csv('test.csv')


# %%
train = train_clean.copy()
test = test_clean.copy()


# %%
train.info()


# %%
print(train.shape)
print(test.shape)


# %%
data = pd.concat([train, test], keys=('x', 'y'))
data


# %%
data.isnull().sum().sort_values(ascending=False)[:20]


# %%
data = data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)


# %%
data.isnull().sum().sort_values(ascending=False)[:20]


# %%
num_data = data._get_numeric_data().columns.tolist()
cat_data = set(data.columns) - set(num_data)


# %%
cat_data


# %%
for col in num_data:
    data[col].fillna(data[col].mean(), inplace=True)
    
for col in cat_data:
    data[col].fillna(data[col].mode()[0], inplace=True)


# %%
for col in cat_data:
    if data[col].value_counts()[0]/data[col].value_counts().sum() > 0.85:
        data = data.drop(col, axis=1)


# %%
cat_data = set(data.columns) - set(num_data)
for col in cat_data:
    print(data[col].value_counts()[0]/data[col].value_counts().sum())


# %%
import seaborn as sns
sns.histplot(data=train, x="SalePrice")
st.pyplot()


# %%
train['SalePrice'] = np.log1p(train['SalePrice'])
data['SalePrice'] = np.log1p(data['SalePrice'])


# %%
sns.histplot(data=train, x="SalePrice")
st.pyplot()


# %%
train.corrwith(train['SalePrice']).sort_values()


# %%
data = data.drop(["PoolArea", "MoSold", "3SsnPorch", "BsmtFinSF2", "BsmtHalfBath",
                  "MiscVal", "LowQualFinSF", "YrSold", "OverallCond", "MSSubClass"],
                 axis = 1)


# %%
data.info()


# %%
data.describe()


# %%
def mod_outliers(data):
    df1 = data.copy()
    data = data[["LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtUnfSF", "TotalBsmtSF",
                "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageArea", "WoodDeckSF",
                "OpenPorchSF"]]
    
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    
    iqr = q3 - q1
    
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    for col in data.columns:
        for i in range(0, len(data[col])):
            if data[col][i] < lower_bound[col]:
                data[col][i] = lower_bound[col]
                
            if data[col][i] > upper_bound[col]:
                data[col][i] = upper_bound[col]
                
    for col in data.columns:
        df1[col] = data[col]
        
    return(df1)


# %%
data = mod_outliers(data)


# %%
data.describe()


# %%
data2 = data.copy()
data = pd.get_dummies(data)


# %%
train = data.loc["x"]
test = data.loc["y"]


# %%
X = train
X.drop('Id', axis=1, inplace=True)
y = X.pop('SalePrice')
