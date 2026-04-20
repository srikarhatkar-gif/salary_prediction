import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
# Assuming 'rf_model.pkl' is in the same directory
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
# Assuming 'label_encoders.pkl' is in the same directory
with open('label_encoders.pkl', 'rb') as file: # Corrected filename here
    label_encoders = pickle.load(file)

st.title('Salary Prediction App')
st.write('Enter the details below to predict the salary.')

# Input fields for user
age = st.slider('Age', 20, 65, 30)
gender_options = ['Male', 'Female']
gender = st.selectbox('Gender', gender_options)
education_options = ['High School', 'Bachelor\'s', 'Master\'s', 'PhD']
education_level = st.selectbox('Education Level', education_options)
job_title = st.text_input('Job Title (e.g., Software Engineer, Data Analyst)')
years_of_experience = st.slider('Years of Experience', 0, 40, 5)

# Map inputs to model's expected numerical format
def preprocess_input(age, gender, education_level, job_title, years_of_experience):
    # Apply Label Encoding using the loaded encoders
    encoded_gender = label_encoders['Gender'].transform([gender])[0]
    encoded_education = label_encoders['Education Level'].transform([education_level])[0]

    # For Job Title, we need to handle cases where a new job title might be entered
    # This is a simplified approach, a more robust solution would involve:
    # 1. Training with more comprehensive job titles
    # 2. Using a hashing trick or a default/most common value if new job title
    #    is encountered at prediction time.
    # For demonstration, we'll try to transform, if not found, use a placeholder.
    try:
        encoded_job_title = label_encoders['Job Title'].transform([job_title])[0]
    except ValueError:
        # If job title is not in the trained categories, assign a default/average value
        # or inform the user. Here, we'll assign 0 (first category) for simplicity.
        # In a real-world scenario, you might want to use a more sophisticated approach
        # like nearest neighbor or embedding.
        encoded_job_title = 0 # Or a more appropriate placeholder
        st.warning(f"Job Title '{job_title}' not seen during training. Using a default encoding.")

    # Create a DataFrame matching the training data's column order
    input_df = pd.DataFrame([[age, encoded_gender, encoded_education, encoded_job_title, years_of_experience]],
                            columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])
    return input_df

if st.button('Predict Salary'):
    try:
        processed_input = preprocess_input(age, gender, education_level, job_title, years_of_experience)
        prediction = model.predict(processed_input)
        st.success(f'Predicted Salary: ${prediction[0]:,.2f}')
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all inputs are valid and try again. For Job Title, try using common titles like 'Software Engineer' or 'Data Analyst'.")

st.caption('Note: This is a simplified model for demonstration purposes and may not reflect real-world salary accuracy.')
!lt --port 8501
