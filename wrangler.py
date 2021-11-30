import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns

st.set_page_config(layout="wide")
col1, col2 = st.columns(2)
st.set_option('deprecation.showPyplotGlobalUse', False)

with col1:
    st.title("Data Wrangling Utility")
    st.write('\n')
    st.write('\n')
    data = None
    dataset_type = st.selectbox('Dataset Type', ('Single File', 'Train & Test files'))
    if dataset_type == 'Single File':
        train_file = st.file_uploader("Upload Train File")
        train_clean = None
        test_clean = None
        if train_file is None:
            st.write('Default data successfully loaded. ')
            train_clean = pd.read_csv('train.csv')
            test_clean = pd.read_csv('test.csv')
            train = train_clean.copy()
            test = test_clean.copy()
            data = pd.concat([train, test], keys=('x', 'y'))
        else:
            train_clean = pd.read_csv(train_file)
            data = train_clean.copy()

    else:
        train_file = st.file_uploader("Upload the training data")
        test_file = st.file_uploader("Upload the test data")
        train_clean = None
        test_clean = None
        
        if train_file is None or test_file is None:
            st.write('Default data successfully loaded. ')
            train_clean = pd.read_csv('train.csv')
            test_clean = pd.read_csv('test.csv')
        else:
            st.write('Uploaded data successfully loaded. ')
            train_clean = pd.read_csv(train_file)
            test_clean = pd.read_csv(test_file)
        
        train = train_clean.copy()
        test = test_clean.copy()
        data = pd.concat([train, test], keys=('x', 'y'))

    targetvariable = st.selectbox('Target Variable', tuple(data.columns))
    
    st.dataframe(data)

    st.header("Null Values")

    data.isnull().sum().sort_values(ascending=False)[:20]
    st.text('''It's a wise idea to drop columns with a large number of null values (>70%)\nas they usually don't contribute much to the model and impact it negatively''')
    st.write("Drop the top n columns")
    n = st.slider('n')
    st.write('Number of columns before dropping: ', len(data.columns))

    for col in data.isnull().sum().sort_values(ascending=False)[:n].index:
        if(col != targetvariable):
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
    st.text('''Removes columns where the most frequent value is present for over %limit of all values i.e.\nIf percent_limit = 85 and the most frequent vlaue is present for more \nthan 85% of values, the column is dropped.''')

    slider1 = st.slider('Percent Limit')

    for col in cat_data:
        if data[col].value_counts()[0]/data[col].value_counts().sum() > (slider1/100):
            data = data.drop(col, axis=1)

    st.write('Number of remaining columns: ', len(data.columns))

    st.header('Target Variable Skew Adjustment')
    st.text('''         Skew is a measure of the asymmetry of the probability distribution of the target variable. \n
    Reducing the skew of the target variable can improve the performance of the model by making training faster and 
    reducing the risk of overfitting.  
    \n
    A log transformation is used to reduce the skew of the target variable drastically.
    \n
    A square root transformation is used to reduce the skew of the target variable moderately.
    ''')

    option2 = st.selectbox('Scaling for SalePrice', ('No Scaling', 'Logarithmic', 'Square Root'))

    if option2 == 'No Scaling':
        data[targetvariable] = data[targetvariable]
    elif option2 == 'Logarithmic':
        data[targetvariable] = np.log1p(data[targetvariable])
    elif option2 == 'Square Root':
        data[targetvariable] = np.sqrt(data[targetvariable])

    sns.histplot(data=data, x=targetvariable)
    st.pyplot()

with col2:
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.write('\n')
    st.header('Correlation Coeffecients with Target Variable')
    st.latex(r'''  r =
  \frac{ \sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}) }{%
        \sqrt{\sum_{i=1}^{n}(x_i-\bar{x})^2}\sqrt{\sum_{i=1}^{n}(y_i-\bar{y})^2}}
    ''')
    st.text('''The Correlation Coeffecienet or more specifically, the Pearson coefficient is a type of \ncorrelation coefficient that represents the relationship between two variables that are\nmeasured on the same interval or ratio scale. It is a measure of the strength of the\nassociation between two continuous variables. ''')
    corrs = data.corrwith(data[targetvariable]).sort_values()
    st.write(corrs)
    corrs = corrs.abs().sort_values()

    n = st.slider('Drop the n least correlated columns')
    st.write('Number of columns before dropping: ', len(data.columns))
    for col in corrs[:n].index:
        if(col != targetvariable):
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

    st.text('Outliers negatively affect model training as they severely change model weights increasing the error.')
    st.latex(r'IQR = Q_3 - Q_1 \\ Q_3 = Third\ Quartile \\ Q_1 = First\ Quartile')
    st.latex(r'\text{Outliers} < {\text{Q1}} - \text{Threshold}\times\text{IQR} \\ or \\ \text{Outliers} > {\text{Q3}} + \text{Threshold}\times\text{IQR}')
    option3 = st.selectbox('Outlier Treatment', ('No Outlier Treatment', 'Modified Outlier Treatment'))
    if option3 == 'No Outlier Treatment':
        data = data
    elif option3 == 'Modified Outlier Treatment':
        threshold = st.slider('Threshold', min_value=1.0, max_value=3.0, value=1.5, step=0.01)
        data = mod_outliers(data, threshold)

    st.write(data.describe())

    st.header('Feature Scaling')

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

    st.text('Normalization is used to improve training time by scaling down values to smaller numbers')


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
    if dataset_type == 'Single File':
        data.to_csv('data.csv')
    else:
        data.loc["x"].to_csv("traindata.csv", index=False)
        data.loc["y"].to_csv("testdata.csv", index=False)
    st.subheader('Dataframe(s) csv created for use.')