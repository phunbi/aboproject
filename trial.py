import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
#import plotly.express as px
#from sklearn.metrics import plot_confusion_metrics plot_roc_curve plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.metrics import plot_confusion_metrics plot_roc_curve plot_precision_recall_curve
def main():
    st.title('Insurance Claim classification App')
    st.sidebar.title('Insurance Prediction')
    st.markdown('Does your building has an Insurance claim?')
    st.sidebar.markdown('Does building has a claim')

    @st.cache(persist=True)
    def load_data():
        data= pd.read_csv('Desktop/dataset/dsn2019/train.csv')
        return data
       
    @st.cache(persist=True)
    def preprocessing(data):
        X=data.iloc[:,[0,3,5]].values
        y=data.iloc[:,-1].values

        le=LabelEncoder()
        y=le.fit_transform(y.flatten())

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
        return X_train,X_test,y_train,y_test,le
    @st.cache(suppress_st_warning=True)
    def decisionTree(X_train,X_test,y_train,y_test):
        tree=DecisionTreeClassifier(max_leaf_nodes=3,random_state=0)
        tree.fit(X_train,y_train)
        y_pred=tree.predict(X_test)
        score=metrics.accuracy_score(y_test,y_pred)*100
        report=classification_report(y_test,y_pred)
        return score,report,tree

    @st.cache(suppress_st_warning=True)
    def neuralNet(X_train,X_test,y_train,y_test):
        scaler=StandardScaler()
        X_train=scaler.transform(X_train)
        X_test=scaler.transform(X_test)
        clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=1)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        score1=metrics.accuracy_score(y_test,y_pred)*100
        report=classification_report(y_test,y_pred)
        return score1,report,clf

    @st.cache(suppress_st_warning=True)
    def knn_classifier(X_train,X_test,y_train,y_test):
        clf=KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        score=metrics.accuracy_score(y_test,y_pred)*100
        report=classification_report(y_test,y_pred)
        return score,report,clf

    def accept_user_data():
        Building_Type=st.text_input('enter the building type:')
        Claim=st.text_input('does the building has a claim:')
        Building_fenced=st.text_input('is the buiding fenced')
        user_prediction_data=np.array([Building_Type,Claim,Building_fenced]).reshape(1,-1)
        return user_prediction_data











        



    

    
    if st.sidebar.checkbox('show raw data',False):
        st.subheader('Insurance data set(classification)')
        st.write(data)
choose_model=st.sidebar.selectbox('choose model',['NONE','Decision Tree','Neural Network','K-Nearest Neighbours'])



