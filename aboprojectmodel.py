import streamlit as st
import joblib
import pandas as pd
from PIL import Image

file = open('newmodel2.joblib','rb')
model1 = joblib.load(file)

#Image=Image.open('b1.png')
st.title('Insurance Claim Prediction')
#st.image(image,use_column_width=True)
#st.subheader('')

html_temp = """
    <div style ='background-color: Aqua; padding:10px'>
    <h2> Aim: To predict if a building will have a claim on Insurance or not.</h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.write('Please provide accurate details for the buildings insured')
YearOfObservation = st.selectbox('YearOfObservation',(0,1))
Insured_Period	= st.slider('Insured_Period	',1.0,1.1)
Residential= st.selectbox('Residential',(0,1))
Building_Painted = st.selectbox('Building_Painted',(0,1))
Building_Fenced = st.selectbox('Building_Fenced	',(0,1))
#Building_Fenced_V = st.selectbox('Building_Fenced_V	',('0','1'))
Settlement = st.selectbox('Settlement',(0,1))
Building_Dimension= st.slider('Building Dimension',300,1405)
Building_Type	 = st.selectbox('Building_Type	',(1,2))

features = {'YearOfObservation':YearOfObservation,
'Insured_Period':Insured_Period,
'Residential':Residential,
'Building_Painted':Building_Painted,
'Building_Fenced':Building_Fenced,
'Settlement':Settlement,
'Building Dimension':Building_Dimension,
'Building_Type':Building_Type
}

if st.button('Submit'):
    data = pd.DataFrame(features,index=[0,1])
    st.write(data)

    prediction = model1.predict(data)
    proba = model1.predict_proba(data)[1]

    if prediction[0] == 0:
        st.error('Building has no Claim')
    else:
        st.success('Building has Claim')
        st.balloons()

    proba_df = pd.DataFrame(proba,columns=['Probability'],index=['Claim','No Claim'])
    proba_df.plot(kind='barh')
    st.pyplot()


