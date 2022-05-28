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
import seaborn as sns
import pandas as pd
import warnings
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




# Title
st.title("Stock Market Dashboard")

# Description

#sidebar
sideBar = st.sidebar
display = sideBar.checkbox('Display Dataset')

uploaded_file = sideBar.file_uploader("Choose a CSV file")


if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):

        df=pd.read_csv(uploaded_file)
        if display:
            st.dataframe(df)


    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
        if display:
            st.dataframe(df)

    else:
        sideBar.write("Please upload csv or excel files only")


