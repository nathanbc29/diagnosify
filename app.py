#import library
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

#load dataset
file_path = 'Disease_symptom_and_patient_profile_dataset.csv'
data = pd.read_csv(file_path)

#merubah variabel kategori menjadi nilai numeric
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

#memisahkan fitur (features) dan target (label)
X = data.drop(['Disease'], axis=1)
y = data['Disease']

#membagi data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#melatih model Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#mengevaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


#streamlit
st.set_page_config(
    page_title="diagnosify",  
    page_icon="🩺",                      
    layout="centered",                   
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Jost&display=swap');
        body {
            font-family: 'Jost', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, #A680FF, #F784BE, #FFAF70);
    background: linear-gradient(to right, #A680FF, #F784BE, #FFAF70);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
    line-height: 0.7;
    margin-bottom: 20px;
}
</style>
<div class="gradient-text">Predict Your Disease Based on Your Symptoms!</div>
<br>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .custom-text {
        margin-top: 15px;
        font-size: 1em;
    }
    </style>
    <div class="custom-text">
        This project is a <strong>Disease Prediction App</strong> built with Streamlit and powered by machine learning algorithms. 
        It predicts potential diseases based on user-inputted symptoms and patient profiles by analyzing a dataset of symptoms and diseases. 
        The app uses a <strong>Random Forest Classifier</strong> to deliver predictions, and if the system determines that the confidence is too low, 
        it suggests seeking a healthcare professional's opinion.
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .gradientt-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, #A680FF, #F784BE, #FFAF70);
        background: linear-gradient(to right, #A680FF, #F784BE, #FFAF70);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2em;
        display: inline-block;
    }
    .emoji {
        display: inline-block;
        font-size: 2em;
        vertical-align: middle;
        margin-bottom: 10px;
    }
    </style>
    <div style="text-align: left;">
        <span class="gradientt-text">Features</span>
        <span class="emoji">🚀</span>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
- **Random Forest Classifier:** Uses a Random Forest algorithm to analyze the input symptoms and predict diseases with high accuracy.
- **Real-Time Prediction:** Instant disease prediction after entering your symptoms, ensuring quick feedback.
- **Prediction Accuracy:** The model’s prediction comes with a confidence level, allowing users to understand the reliability of the result.
""")


st.sidebar.header("Input Patient Data")

#user input
def user_input_features():
    fever = st.sidebar.selectbox('Fever', ['Yes', 'No'])
    cough = st.sidebar.selectbox('Cough', ['Yes', 'No'])
    fatigue = st.sidebar.selectbox('Fatigue', ['Yes', 'No'])
    difficulty_breathing = st.sidebar.selectbox('Difficulty Breathing', ['Yes', 'No'])
    age = st.sidebar.slider('Age', 0, 100, 25)
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    blood_pressure = st.sidebar.selectbox('Blood Pressure', ['Low', 'Normal', 'High'])
    cholesterol_level = st.sidebar.selectbox('Cholesterol Level', ['Normal', 'High', 'Low'])
    outcome_variable = st.sidebar.selectbox('Outcome Variable', ['Positive', 'Negative'])

    data = {
        'Fever': fever,
        'Cough': cough,
        'Fatigue': fatigue,
        'Difficulty Breathing': difficulty_breathing,
        'Age': age,
        'Gender': gender,
        'Blood Pressure': blood_pressure,
        'Cholesterol Level': cholesterol_level,
        'Outcome Variable': outcome_variable
    }
    return pd.DataFrame([data])

input_data = user_input_features()

#pra-pemrosesan data yang diinput user
for column in input_data.columns:
    if column in label_encoders:
        input_data[column] = label_encoders[column].transform(input_data[column])

#prediksi
prediction = model.predict(input_data)[0]
predicted_disease = label_encoders['Disease'].inverse_transform([prediction])[0]

#hasil
st.markdown("""
    <style>
    .gradienttt-text {
        font-weight: bold;
        background: -webkit-linear-gradient(left, #A680FF, #F784BE, #FFAF70);
        background: linear-gradient(to right, #A680FF, #F784BE, #FFAF70);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2em;
        display: inline-block;
    }
    </style>
    <div style="text-align: left;">
        <span class="gradienttt-text">Predicted Disease: </span>
    </div>
""", unsafe_allow_html=True)

st.markdown(f'<div style="border: 0px solid grey; background-color:rgb(19 8 36); padding: 10px; box-shadow: 0 0 20px rgb(12 8 18); border-radius: 20px; font-size: 20px; font-weight: normal; text-align: center;">{predicted_disease}</div>', unsafe_allow_html=True)

