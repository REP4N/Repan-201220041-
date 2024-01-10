import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load Naive Bayes model
naive_bayes_model = pickle.load(open('naive_bayes_model.pkl', 'rb'))

# Load dataset
data = pd.read_csv('Bank Customer Churn Dataset.csv')

st.title('Aplikasi Bank Customer')

html_layout1 = """
<br>
<div style="background-color:blue ; padding:2px">
<h2 style="color:white;text-align:center;font-size:40px"><b>Pelanggan Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1, unsafe_allow_html=True)

activities = ['Naive Bayes', 'Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?', activities)
st.sidebar.header('Data Pelanggan')

if st.checkbox("Tentang Dataset"):
    html_layout2 = """
    <br>
    <p>Ini adalah dataset Bank Customer</p>
    """
    st.markdown(html_layout2, unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

if st.checkbox('EDA'):
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')

# Train-test split
X = data.drop('churn', axis=1)
y = data['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Separate categorical and numeric variables
X_categorical = X[['country', 'gender']]
X_numeric = X.drop(['country', 'gender'], axis=1)

# One-hot encoding on categorical variables
# Combine the one-hot encoded categorical variables with numeric variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), X_numeric.columns),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), X_categorical.columns)
    ])

X_encoded = pd.DataFrame(preprocessor.fit_transform(X))

# Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    credit_score = st.sidebar.slider('Credit Score', 0, 200, 108)
    country = st.sidebar.selectbox('Country', data['country'].unique())
    gender = st.sidebar.selectbox('Gender', data['gender'].unique())
    age = st.sidebar.slider('Age', 21, 100, 24, step=1)
    tenure = st.sidebar.slider('Tenure', 0, 80, 25)
    
    balance = st.sidebar.number_input('Balance', min_value=0.0, max_value=2.5, step=0.01, value=0.45)
    
    products_number = st.sidebar.slider('Products Number', 21, 100, 24, step=1)
    credit_card = st.sidebar.slider('Credit Card', 0, 100, 24, step=1)
    active_member = st.sidebar.slider('Active Member', 0, 100, 24, step=1)
    estimated_salary = st.sidebar.slider('Estimated Salary', 0, 100, 24, step=1)

    user_report_data = {
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }

    # Convert the user_report_data dictionary to a DataFrame
    report_df = pd.DataFrame([user_report_data])

    # Tambahkan 'customer_id' ke dalam DataFrame report_df
    report_df['customer_id'] = report_df.index

    # Define used_features here
    used_features = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'customer_id']

    # Convert list to tuple
    used_features_tuple = tuple(used_features)

    # Transform the user data using the preprocessor
    report_data = pd.DataFrame(preprocessor.transform(report_df[used_features]), columns=preprocessor.get_feature_names_out())

    return report_data.to_dict(orient='records')[0]

# User Data
user_data = user_report()
st.subheader('Data Pelangan')
st.write(user_data)

# Adjust features used during prediction with features used during model training
# Select only the features used during model training
used_features = ['credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'products_number', 'credit_card', 'active_member', 'estimated_salary', 'customer_id']
user_result = naive_bayes_model.predict(pd.DataFrame([user_data[used_features]], columns=used_features))
user_accuracy = accuracy_score([user_result[0]], [1])

# Output
st.subheader('Hasil Prediksi:')
if user_result[0] == 0:
    output = 'Pelanggan tetap (churn negatif)'
else:
    output = 'Pelanggan berpindah (churn positif)'
st.title(output)
st.subheader('Model yang digunakan : \n' + option)
st.subheader('Accuracy : ')
st.write(str(user_accuracy * 100) + '%')
