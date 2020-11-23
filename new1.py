import streamlit as st
#import joblib
import pandas as pd
import numpy as np

#file = open('model1.joblib','rb')
#model1 = joblib.load(file)

from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

#import the data
train=pd.read_csv('train.csv')
Image=Image.open('house.png')
st.title('Welcome to the Insurance Prediction App')
st.image(image,use_column_width=True)

#checking the data
st.write('This is the Application for knowing the Houses with or without Claim using machine learning')
check_train=st.checkbox('see the simple data')
if check_train:
    st.write(train.head())
    st.write('Now lets check which of the buildings are with claim or without claim.')

#input the numbers
    Residential=st.slider('Is the area residential?',int(train.Residential.min()),int(train.Residential.mean()))
    Building_Type=st.slider('What kind of building?',int(train.Building_Type.min()),int(train.Building_Type.mean()))
    YearOfObservation=st.slider('What year did you get the building?',int(train.YearOfObservation.min()),int(train.YearOfObservation.mean()))
    Building_Painted=st.slider('Is the Building painted?',object(train.Building_Painted.min()),object(train.Building_Painted.mean()))
#splitting the data
X=train.drop('Claim',axis=1)
y=train['Claim']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=.2,random_state=45

#modelling step
#import model

#fitting and predict your model
model =LinearRegression()
model.fit(X_train, y_train)
model.predict(X_test)
errors=np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
predictions=model.predict([[Residential,Building_Type,YearOfObservation]])[0]

#checking predictions
if st.button('Run me!'):
    st.header('your house has a claim {}'.format(int(predictions)))
    st.subheader('your house has no claim {}'.format(int(predictions-errors),int(predictions+errors)))