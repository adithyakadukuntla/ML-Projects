# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:47:14 2024

@author: Lenovo
"""

import pickle
import pandas as pd
import numpy as np
import streamlit as st
import difflib
import warnings
warnings.filterwarnings('ignore')
from streamlit_option_menu import option_menu


### Diabetes Prediction
diabetes_sav = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/1.Diabetes_prediction/trained_diabetes_model.sav",'rb'));
diabetes_scaler = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/1.Diabetes_prediction/scaler.sav",'rb'))

### Loan Sanction Prediction
loan_sav = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/2.Loan_Prediction/Loan_status_predictor.sav",'rb'))
loan_scaler = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/2.Loan_Prediction/Loan_status_scaler.sav",'rb'))

## Heart Failure 
heart_sav = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/3.Heart_Failure_prediction/Heart_Disease_pred.sav",'rb'))
heart_scaler = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/3.Heart_Failure_prediction/Heart_scaler.sav",'rb'))

### Medical Insurance 
medical_insurance_sav = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/4.Medical_insurance/Insurance_model.sav",'rb'))

## Movie recommended system
df = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/5.Movie_Recommended/movies_Recom.csv')
movie_similarity = pickle.load(open('C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/5.Movie_Recommended/movie_similarity.sav','rb'))
list_of_movies = pickle.load(open('C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/5.Movie_Recommended/all_movies.sav','rb'))

## Parkinsons prdiction
pd_scaler = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/6.Parkinsons's_Disease_Pred/scaler_parkinsons.sav",'rb'))
pd_model = pickle.load(open("C:/Users/Lenovo/OneDrive/Desktop/today/Machine_Learning/6.Parkinsons's_Disease_Pred/parkinsons_LOR_model.sav",'rb'))


with st.sidebar:
    
    selected = option_menu('Machine Learnig Projects',
                           ['Diabetes Prediction','Loan Status Prediction','Heart Failure Prediction','Medical Insurance Predict','Movie Recommended System',"Parkinsons's Disease Prediction" ],
                           default_index=0)
    
## Diabetes
def diabetes_prediction(input_data):
    

    # changing the input data to numpy array

    ip_np = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    id_reshape = ip_np.reshape(1,-1)
    
    std_data = diabetes_scaler.transform(id_reshape)


    prediction = diabetes_sav.predict(std_data)

    print('prediction',prediction)

    if (prediction[0]==0):
      return'The person is not diabetic'
    else:
      return'The person is diabetic'
      
if(selected=='Diabetes Prediction'):
    
    st.title('Diabetes Prediction')
    Pregencies = st.text_input("Number of Pregencies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("BloodPressure value")
    SkinThickness = st.text_input("SkinThickness")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction ")
    Age = st.text_input("Age of person")
    
    
    #code for prediction
    diagonis = ''
    
    #creating a button
    
    if st.button('Diabetes Test Result'):
        diagonis= diabetes_prediction([Pregencies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    
    st.success(diagonis)
    
    
##Loan sanction
def Loan_Prediction(input_data):
    ip_np = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    id_reshape = ip_np.reshape(1, -1)

    # Standardizing the values of input_data
    std_data1 = loan_scaler.transform(id_reshape)

    # Predict the loan status
    prediction = loan_sav.predict(std_data1)

    if prediction[0] == 'Y':
        return 'The loan is sanctioned'
    else:
        return 'The loan is not sanctioned'

if(selected=='Loan Status Prediction'):
    
    st.title('Loan Status Prediction')
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11 = st.columns(2)

    with col1:
        Gender = st.text_input('Gender 1-Male 0-Female ')
    with col2:
        Married = st.text_input('Married 1-Yes 0-NO')
    with col3:
        Dependents = st.text_input('Dependents 0,1,2,3')

    with col4:
        Education = st.text_input('Education 1-Graduated 0-Not Graduated')
    with col5:
        Self_Employed = st.text_input('Self Employed  1-Yes 0-No')
    with col6:
        ApplicantIncome = st.text_input('ApplicantIncome')

    with col7:
        CoapplicantIncome = st.text_input('CoapplicantIncome')
    with col8:
        LoanAmount = st.text_input('LoanAmount')
    with col9:
        Loan_Amount_Term = st.text_input('Loan_Amount_Term')

    with col10:
        Credit_History = st.text_input('Credit_History')
    with col11:
        Property_Area = st.text_input('Property_Area  0-Rural 1-Semiurban 2-Urban')

    Status = ''

    if st.button('Check Status'):
        Status = Loan_Prediction([Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area])

    if (Status=='The loan is sanctioned'):
        st.success(Status)
    if (Status=='The loan is not sanctioned'):
        st.warning(Status)

## Heart Disease
def Heart_Disease(input_data):
    
    ip_np = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    id_reshape = ip_np.reshape(1,-1)


    ### standardizing the values of inpu_data
    ## bcoz we are standardized the csv data so that they can prdict nicely

    std_data1 =heart_scaler.transform(id_reshape)

    

    prediction = heart_sav.predict(std_data1)

    #print('prediction',prediction)

    if (prediction[0]==0):
      return'The Heart of person is NORMAL'
    else:
      return 'The Heart of this person has HEART DISEASE'
    
if(selected=='Heart Failure Prediction'):
    
    st.title('Heart Failure Prediction')
    
    col1,col2,col3 = st.columns(3)
    col4,col5,col6 = st.columns(3)
    col7,col8,col9 = st.columns(3)
    col10,col11 = st.columns(2)
    # Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
    with col1:
        Age = st.text_input('Age')
    with col2:
        Sex = st.text_input('Sex 1-Male 0-Female')
    with col3:
        ChestPainType_value = st.selectbox('Chest Pain Type', ['TA: Typical Angina', 'ATA: Atypical Angina', 'NAP: Non-Anginal Pain', 'ASY: Asymptomatic'], index=0)
    
    Chest_pain_map = {'TA: Typical Angina': 0, 'ATA: Atypical Angina': 1, 'NAP: Non-Anginal Pain': 2, 'ASY: Asymptomatic': 3}
    ChestPainType =Chest_pain_map[ChestPainType_value]
    
    with col4:
        RestingBP = st.text_input('RestingBP [mm Hg]')
    with col5:
        Cholesterol = st.text_input('serum Cholesterol [mm/dl]')
    with col6:
        FastingBS = st.text_input('FastingBS  [1: if FastingBS > 120 mg/dl, 0: otherwise]')

    with col7:
        RestingECG_values = st.selectbox('RestingECG', ['Normal: Normal', 'ST: having ST-T wave abnormality (>0.05 mV)','LVH: showing probable'], index=0)
        
    RestingECG_map ={'Normal: Normal':1 , 'ST: having ST-T wave abnormality (>0.05 mV)':0 , 'LVH: showing probable':2}
    RestingECG = RestingECG_map[RestingECG_values]
        
    with col8:
        MaxHR = st.text_input('Maximum heart rate achieved [Numeric value between 60 and 202]')
    with col9:
        ExerciseAngina_values = st.selectbox('ExerciseAngina',['Yes','No'])
        
    ExerciseAngina_map = {'Yes':1 ,'No':0}
    ExerciseAngina = ExerciseAngina_map[ExerciseAngina_values]

    with col10:
        Oldpeak = st.text_input('Oldpeak ST [Numeric value measured in depression]')
    with col11:
        ST_Slope_values = st.selectbox('ST_Slope',['Up: upsloping', 'Flat: flat', 'Down: downsloping'])
        
    ST_Slope_map = {'Up: upsloping':2, 'Flat: flat':1, 'Down: downsloping':0}
    ST_Slope = ST_Slope_map[ST_Slope_values]
    
    Status = ''
    
    if st.button("PREDICT"):
        Status = Heart_Disease([Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
        
    if (Status=='The Heart of person is NORMAL'):
        st.success(Status)
    if (Status=='The Heart of this person has HEART DISEASE'):
        st.warning(Status)
    
##Medical Insurance
def Medical_insurance(input_data):
    ip_np = np.asarray(input_data)

#reshape the array as we are predicting for one instance
    id_reshape = ip_np.reshape(1,-1)


    prediction = medical_insurance_sav.predict(id_reshape)
    ans = str(prediction[0])
    print(ans)
    return ans

if(selected == 'Medical Insurance Predict'):
    
    st.title('Medical Insurance Predict')
    
    age = st.text_input('Age')
    sex = st.text_input(' Gender : male - 1 or female - 0')
    bmi = st.text_input('BMI')
    children = st.text_input('No of children')
    smoker = st.text_input('Smoker ? Yes -1 or No - 0')
 ##region_box = st.selectbox('Region',['southwest','southeast','northwest','northeast'])
   #region_map = {'southeast':0,'southwest':1,'northeast':2,'northwest':3}
    region = st.text_input("'southeast':0,'southwest':1,'northeast':2,'northwest':3")
    
    
    
    result = ''
    if st.button('Predict'):
        age = int(age)
        sex = int(sex)
        bmi = float(bmi)
        children = int(children)
        smoker = int(smoker)
        region = int(region)
        result = Medical_insurance([age,sex,bmi,children,smoker,region])
        
    st.success(result)
    
    
## Movie Recommended system
def Recomend_Movie(input_data):
    
    movie_name = input_data

    find_close_match = difflib.get_close_matches(movie_name,list_of_movies)

    close_matched = find_close_match[0]

    index_of_movie  = df[df.title==close_matched]['index'].values[0]

    similarity_score = list(enumerate(movie_similarity[index_of_movie]))

    sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True) 
    # print the name of similar movies based on the index

    

    i = 1
    movies_list =[]
    for movie in sorted_similar_movies:
      index = movie[0]
      
      title_from_index = df[df.index==index]['title'].values[0]
      if (i<11):
        #print(i, '.',title_from_index)
        movies_list.append(title_from_index)
        i+=1
    return movies_list

if(selected == 'Movie Recommended System'):
    
    st.title('Movie Recommended System')
    
    movie = st.text_input('Movie name : ')
    
    
    movies= []
    
    
    if st.button('Recommend Movies'):
        movies = Recomend_Movie(movie)
        
    if len(movies) > 0:
        st.success('Movie Recommended for You:')
        st.markdown('\n'.join(f"- {movie}" for movie in movies))
        

## Parkinsons disease prediction
def Parkinson_prediction(input_data):
    
    ip_np = np.asarray(input_data)

    #reshape the array as we are predicting for one instance
    id_reshape = ip_np.reshape(1,-1)
    
    
    ### standardizing the values of inpu_data
    ## bcoz we are standardized the csv data so that they can prdict nicely
    
    std_data1 =pd_scaler.transform(id_reshape)
    
    
    
    prediction = pd_model.predict(std_data1)
    
    
    
    if (prediction[0]==1):
      return 'The person has parkinson disease'
    else:
      return 'The person doesnt have parkinsons disease'
      
if(selected == "Parkinsons's Disease Prediction"):
    
    st.title("Parkinsons's Disease Prediction")
    
    #name,,,,,,,,,,,,,,,status,,,,,,
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    col7, col8, col9 = st.columns(3)
    col10, col11,col12 = st.columns(3)
    col13, col14,col15 = st.columns(3)
    col16, col17,col18 = st.columns(3)
    col19, col20,col21 = st.columns(3)
    
    
    
    with col1:
        MDVPFo_hz = st.text_input('MDVP_Fo_Hz')
    with col2:
        MDVPFhi_hz = st.text_input('MDVP_Fhi_Hz')
    with col3:
        MDVPFlo_hz = st.text_input('MDVP_Flo_Hz')
        
    with col4:
        MDVPJitter_per= st.text_input('MDVP_Jitter_per')
    with col5 :
        MDVPJitter_abs = st.text_input('MDVP_Jitter_Abs')
    with col6:
        MDVPRAP = st.text_input('MDVP_RAP')
    with col7:
        MDVPPPQ = st.text_input('MDVP_PPQ')
    with col8:
        JitterDDP = st.text_input('Jitter_DDP')
    with col9:
        MDVPShimmer = st.text_input('MDVP_Shimmer')
    with col10:
        MDVPShimmer_db = st.text_input('MDVP_Shimmer_db')
    with col11:
        ShimmerAPQ3 = st.text_input('Shimmer_APQ3')
    with col12:
        ShimmerAPQ5 =st.text_input('Shimmer_APQ5')
    with col13:
        MDVPAPQ = st.text_input('MDVP_APQ')
    with col14:
        ShimmerDDA = st.text_input('Shimmer_DDA')
    with col15:
        NHR = st.text_input('NHR')
    with col16:
        HNR = st.text_input('HNR')
    with col17:
        RPDE = st.text_input('RPDE')
    with col18:
        DFA = st.text_input('DFA')
    with col19:
        spread1 =st.text_input('spread1')
    with col20:
        spread2 = st.text_input('spread2')
    with col21:
        D2 =st.text_input('D2')
    
    PPE = st.text_input('PPE')
        
    result=''
    if st.button('Predict'):
        result = Parkinson_prediction([MDVPFo_hz,MDVPFhi_hz,MDVPFlo_hz,MDVPJitter_per,MDVPJitter_abs,MDVPRAP,MDVPPPQ,JitterDDP,MDVPShimmer,MDVPShimmer_db,ShimmerAPQ3,ShimmerAPQ5,MDVPAPQ,ShimmerDDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])

    st.success(result)        
        
        
    
    
    
    
    
    
    
    
    
    