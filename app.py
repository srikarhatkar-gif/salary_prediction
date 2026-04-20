import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load model and encoders
# -------------------------------
@st.cache_resource
def load_model():
    with open('rf_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

model = load_model()
label_encoders = load_encoders()

# -------------------------------
# Load dataset for dropdown
# -------------------------------
@st.cache_data
def load_job_titles():
    df = pd.read_csv('Salary_Data.csv')
    job_titles = sorted(df['Job Title'].dropna().unique())
    return job_titles

job_title_list = load_job_titles()

# -------------------------------
# App UI
# -------------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💰")

st.title("💰 Salary Prediction App")
st.write("Enter the details below to predict salary.")

# Inputs
age = st.slider('Age', 20, 65, 30)

gender = st.selectbox('Gender', ['Male', 'Female'])

education_level = st.selectbox(
    'Education Level',
    ['High School', "Bachelor's", "Master's", 'PhD']
)

job_title = st.selectbox('Job Title', job_title_list)

years_of_experience = st.slider('Years of Experience', 0, 40, 5)

# -------------------------------
# Preprocessing
# -------------------------------
def preprocess_input(age, gender, education_level, job_title, years):
    try:
        encoded_gender = label_encoders['Gender'].transform([gender])[0]
        encoded_education = label_encoders['Education Level'].transform([education_level])[0]
        encoded_job_title = label_encoders['Job Title'].transform([job_title])[0]
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return None

    input_df = pd.DataFrame(
        [[age, encoded_gender, encoded_education, encoded_job_title, years]],
        columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience']
    )

    return input_df

# -------------------------------
# Prediction
# -------------------------------
if st.button('Predict Salary'):
    processed_input = preprocess_input(
        age, gender, education_level, job_title, years_of_experience
    )

    if processed_input is not None:
        try:
            prediction = model.predict(processed_input)[0]
            st.success(f"💵 Predicted Salary: INR{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("⚠️ This model is for educational purposes and may not reflect real-world salaries.")
