import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title('Model Building')

train = pd.read_csv('traindata.csv')
test = pd.read_csv('testdata.csv')
st.write('Cleaned Data has been successfully loaded')
st.write(train.head())



X = train
X.drop('Id', axis=1, inplace=True)
y = X.pop('SalePrice')

testid = test.Id
test.pop('SalePrice')
test.pop('Id')

print(train.shape)
print(test.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42)


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def rmse_cv(model, trainset):
    rmse = np.sqrt(-cross_val_score(model, trainset, y_train, scoring = "neg_mean_squared_error",
                                    cv = 3))
    return(rmse)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

option1 = st.radio('Select a model', ('Linear Regression', 'Random Forest', 
'XGBoost', 'Kernel Ridge', 'Elastic Net', 'Bayesian Ridge', 'Gradient Boosting', 'SVR'))

tune = st.checkbox('Tune Hyperparameters')


if option1 == 'Linear Regression':
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'Random Forest':
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'XGBoost':
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # y_mainpred = model.predict(test)
    # my_submission = pd.DataFrame({'Id': testid, 'SalePrice': y_mainpred})
    # my_submission.to_csv('submission.csv', index=False)
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'Kernel Ridge':
    param_grid = {"alpha": [1e-1, 1, 1e2], "gamma": [None,-2, 2, 5]}
    if tune:
        model = GridSearchCV(KernelRidge(),param_grid,cv=5)
        model.fit(X_train, y_train)
        y_pred = model.best_estimator_.predict(X_test)
    else:
        model = KernelRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'Elastic Net':
    param_grid = {"alpha": [1e-1, 1e1, 1e2], "l1_ratio": [0.1, 0.5, 1]}
    if tune:
        model = GridSearchCV(ElasticNet(),param_grid,cv=5)
        model.fit(X_train, y_train)
        y_pred = model.best_estimator_.predict(X_test)
    else:
        model = ElasticNet()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'Bayesian Ridge':
    param_grid = {"alpha_1": [1e-6, 1e-7, 1e-5], "alpha_2": [1e-5, 1e-6, 1e-7]}
    if tune:
        model = GridSearchCV(BayesianRidge(),param_grid,cv=5)
        model.fit(X_train, y_train)
        y_pred = model.best_estimator_.predict(X_test)
    else:
        model = BayesianRidge()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'Gradient Boosting':
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))
elif option1 == 'SVR':
    param_grid = {"C": [1e1, 1e2, 1e3], "gamma": ['scale', 1e-3, 1e-2, 1e-1]}
    if tune:
        model = GridSearchCV(SVR(),param_grid,cv=5)
        model.fit(X_train, y_train)
        y_pred = model.best_estimator_.predict(X_test)
        # y_mainpred = model.best_estimator_.predict(test)
        # my_submission = pd.DataFrame({'Id': testid, 'SalePrice': y_mainpred})
        # my_submission.to_csv('submission.csv', index=False)
        # my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
    else:
        model = SVR()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write('RMSE: ', rmse)
    st.write('Cross Validation Score: ', rmse_cv(model, X_train))


