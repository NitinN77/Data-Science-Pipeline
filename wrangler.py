import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
st.set_page_config(layout="wide")
col1, col2 = st.columns(2)
st.set_option('deprecation.showPyplotGlobalUse', False)

with col1:
    st.title("Data Wrangling Utility")
    st.write('\n')
    st.write('\n')
    train_clean = pd.read_csv('train.csv')
    test_clean = pd.read_csv('test.csv')
    train = train_clean.copy()
    test = test_clean.copy()

    data = pd.concat([train, test], keys=('x', 'y'))
    st.write('Data successfully loaded. ')
    st.dataframe(data)

    st.header("Null Values")

    data.isnull().sum().sort_values(ascending=False)[:20]

    st.write("Drop the top n columns")
    n = st.slider('n')
    st.write('Number of columns before dropping: ', len(data.columns))

    for col in data.isnull().sum().sort_values(ascending=False)[:n].index:
        if(col != 'SalePrice' and col != 'Id'):
            data = data.drop(col, axis=1)

    st.write('Number of columns after dropping', len(data.columns))
    num_data = data._get_numeric_data().columns.tolist()
    cat_data = set(data.columns) - set(num_data)

    option1 = st.selectbox('Replace remaining numeric null values with: ', 
    ('Mean', 'Median'))

    if option1 == 'Mean':
        for col in num_data:
            data[col].fillna(data[col].mean(), inplace=True)
    else:
        for col in num_data:
            data[col].fillna(data[col].median(), inplace=True)
        
    for col in cat_data:
        data[col].fillna(data[col].mode()[0], inplace=True)

    st.header('Low Variance Filtering')

    slider1 = st.slider('Percent Limit')

    for col in cat_data:
        if data[col].value_counts()[0]/data[col].value_counts().sum() > (slider1/100):
            data = data.drop(col, axis=1)

    st.write('Number of remaining columns: ', len(data.columns))

    st.header('SalePrice Skew Adjustment')

    option2 = st.selectbox('Scaling for SalePrice', ('No Scaling', 'Logarithmic', 'Square Root'))

    if option2 == 'No Scaling':
        data['SalePrice'] = data['SalePrice']
    elif option2 == 'Logarithmic':
        train['SalePrice'] = np.log1p(train['SalePrice'])
        data['SalePrice'] = np.log1p(data['SalePrice'])
    elif option2 == 'Sqrt':
        train['SalePrice'] = np.sqrt(train['SalePrice'])
        data['SalePrice'] = np.sqrt(data['SalePrice'])

    sns.histplot(data=train, x="SalePrice")
    st.pyplot()

with col2:
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.header('Correlation Coeffecients with SalePrice')
    corrs = train.corrwith(train['SalePrice']).sort_values()
    st.write(corrs)
    corrs = corrs.abs().sort_values()

    n = st.slider('Drop the n least correlated columns')
    st.write('Number of columns before dropping: ', len(data.columns))
    for col in corrs[:n].index:
        if(col != 'SalePrice' and col != 'Id'):
            data = data.drop(col, axis=1)
    st.write('Number of columns after dropping: ', len(data.columns))

    def mod_outliers(data, threshold):
        df1 = data.copy()
        num_data = data._get_numeric_data().columns.tolist()
        data = data[num_data]
        
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        
        iqr = q3 - q1
        
        lower_bound = q1 - (threshold * iqr)
        upper_bound = q3 + (threshold * iqr)
        
        for col in data.columns:
            for i in range(0, len(data[col])):
                if data[col][i] < lower_bound[col]:
                    data[col][i] = lower_bound[col]
                    
                if data[col][i] > upper_bound[col]:
                    data[col][i] = upper_bound[col]
                    
        for col in data.columns:
            df1[col] = data[col]
            
        return(df1)


    option3 = st.selectbox('Outlier Treatment', ('No Outlier Treatment', 'Modified Outlier Treatment'))
    if option3 == 'No Outlier Treatment':
        data = data
    elif option3 == 'Modified Outlier Treatment':
        threshold = st.slider('Threshold', min_value=1.0, max_value=3.0, value=1.5, step=0.01)
        data = mod_outliers(data, threshold)

    st.write(data.describe())

    st.header('Feature Scaling')

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer


    method = st.selectbox('Scaling Method', ('No Scaling', 'Standardization', 'Min-Max',
    'Max Abs Scaler', 'Quantile Transformer Scaler', 'Power Transformer'))

    num_data = data._get_numeric_data().columns.tolist()
    if method == 'Standardization':
        scaler = StandardScaler()
        data[num_data] = scaler.fit_transform(data[num_data])
    elif method == 'Min-Max':
        scaler = MinMaxScaler()
        data[num_data] = scaler.fit_transform(data[num_data])
    elif method == 'Max Abs Scaler':
        scaler = MaxAbsScaler()
        data[num_data] = scaler.fit_transform(data[num_data])
    elif method == 'Quantile Transformer':
        scaler = QuantileTransformer()
        data[num_data] = scaler.fit_transform(data[num_data])
    elif method == 'Power Transformer':
        scaler = PowerTransformer(method='yeo-johnson')
        data[num_data] = scaler.fit_transform(data[num_data])


    st.write(data.describe())

    data2 = data.copy()
    data = pd.get_dummies(data)
    data.loc["x"].to_csv("traindata.csv", index=False)
    data.loc["y"].to_csv("testdata.csv", index=False)
    st.subheader('Dataframe csv created for use.')