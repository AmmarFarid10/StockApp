import streamlit as st
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from  sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import seaborn as sns
import pandas as pd
import warnings
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore", category=FutureWarning)
st.set_option('deprecation.showPyplotGlobalUse', False)


from io import StringIO

def getClassifier(classifier):
    if classifier == 'SVM':
        c = st.sidebar.slider(label='Choose value of C' , min_value=0.0001, max_value=10.0)
        model = SVC(C=c)
    elif classifier == 'KNN':
        neighbors = st.sidebar.slider(label='Choose Number of Neighbors',min_value=1,max_value=20)
        model = KNeighborsClassifier(n_neighbors = neighbors)
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 10)
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        model = RandomForestClassifier(max_depth = max_depth , n_estimators= n_estimators,random_state= 1)
    return model


def SVM(df):
    df.index = pd.to_datetime(df['Date'])  
        # drop The original date column
    df = df.drop(['Date'], axis='columns')
        # Create predictor variables
    df['Open-Close'] = df.Open - df.Close
    df['High-Low'] = df.High - df.Low
        # Store all predictor variables in a variable X
    X = df[['Open-Close', 'High-Low']]
        # Target variables
    y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    split_percentage = 0.8
    split = int(split_percentage*len(df))
  
        # Train data set
    X_train = X[:split]
    y_train = y[:split]
  
        # Test data set
    X_test = X[split:]
    y_test = y[split:]


        # Support vector classifier
    cls = SVC().fit(X_train, y_train)
    df['Predicted_Signal'] = cls.predict(X)
        # Calculate daily returns
    df['Return'] = df.Close.pct_change()

        # Calculate strategy returns
    df['Strategy_Return'] = df.Return *df.Predicted_Signal.shift(1)
        # Calculate Cumulutive returns
    df['Cum_Ret'] = df['Return'].cumsum()

        # Plot Strategy Cumulative returns 
    df['Cum_Strategy'] = df['Strategy_Return'].cumsum()
    plt.style.use('seaborn-darkgrid')
    df=df.dropna()

    fig,ax=plt.subplots()
    ax.plot(df['Cum_Ret'],color='red')
    ax.plot(df['Cum_Strategy'],color='blue')

    st.pyplot(fig)


# Title
st.title("Stock Market Dashboard")

# Description

#sidebar
sideBar = st.sidebar
display = sideBar.checkbox('Display Dataset')

uploaded_file = sideBar.file_uploader("Choose a CSV file")
classifier = sideBar.selectbox('Which Classifier do you want to use?',('SVM' , 'KNN' , 'Random Forest'))


if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):

        df=pd.read_csv(uploaded_file)
        if display:
            st.dataframe(df)
        if classifier == 'SVM':
            SVM(df)


  




    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        if display:
            st.dataframe(df)
        if classifier == 'SVM':
            SVM(df)

    else:
        sideBar.write("Please upload csv or excel files only")


