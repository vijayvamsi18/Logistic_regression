import streamlit as st
import pickle
import pandas as pd


st.title('Titanic survival prediction')
def get_features():
    st.sidebar.title('enter passenger details')
    pclass=st.sidebar.selectbox('passenger class',[1,2,3])
    sex=st.sidebar.selectbox('Sex',[0,1])
    age=st.sidebar.slider('age',min_value=0,max_value=100)
    sibsp=st.sidebar.slider('siblings/spouses Abroad',0,8,0)
    parch=st.sidebar.slider('parents/children Abroad',0,6,0)
    fare=st.sidebar.number_input('Fare paid')
    embarked=st.sidebar.selectbox('Port of Embarkation',[0,1,2])
    data={'Pclass':pclass,
    'Sex':sex,
    'Age':age,
    'SibSp':sibsp,
    'Parch':parch,
    'Fare':fare,
    'Embarked':embarked}
    features=pd.DataFrame(data,index=[0])
    return features
x_vals=get_features()
if st.sidebar.button('submit'):
    st.write(x_vals)
    # load the model
    loaded_model=pickle.load(open('aclf.pkl','rb'))
    res=loaded_model.predict(x_vals)
    if res == 0:
        st.write('Not Survived')
    else:
        st.write('Survived')