import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import joblib
import sklearn


print(sklearn.__version__) # scikit-learn ver: 1.3.1
print(joblib.__version__)

# Create a function named load_model that takes a model name as input and returns the loaded model.
def load_model(model_name):
    file_path = model_name + ".joblib"  
    loaded_model = joblib.load(file_path)
    return loaded_model

# Load the trained model(s)
model_insomnia = load_model('insomnia_models')
model_anxiety = load_model('anxiety_models')

# Streamlit app
st.title("Integrating Ensemble Machine Learning for Enhanced Multicultural Music Therapy Recommendation Logic in a Music Recommendation System")

st.header("Introduction")
st.write("This app takes in your user inputs and creates predicted levels of self-reported Insomnia and Anxiety by using the model built by the developer (Chi Le), then a music recommendation logic was integrated to display recommended music/playlist that are helpful for your mental health conditions.")
st.markdown("<b>Step-by-step guide: </b>", unsafe_allow_html=True)
st.markdown(
"""
- Toggle the user input bars to provide your information
- The model will display the predicted level of self-reported Insomnia and Anxiety
- The music recommendation logic will display playlist according to your conditions if predicted level is higther than 5 (on a scale of 1-10).
"""
)
st.markdown("<b>Keywords: </b> Ensemble Machine Learning, Bootstrap Aggregation, Music Recommendation System, Music Therapy, Self-music therapy, Multicultural Music Therapy.", unsafe_allow_html=True)

st.header("Ensemble models")
st.write("There are 2 ensemble models. First is the Insomnia Model, the second one is the Anxiety Model. Each model has these 4 algorithms as base models: Random Forest, Gradient Boosting, Support Vector Machine and Neural Networks. Bootstrap aggregation is applied to calculate the mean for these ensemble predictions at the end and evaluated with Mean Square Error (MSE) metrics.

# User inputs for anxiety and insomnia levels
st.sidebar.header("User Inputs")
age = st.sidebar.slider("Age", 18, 100, 30)
hours_per_day = st.sidebar.slider("Hours per day", 0, 24, 8)
depression = st.sidebar.slider("Depression Level (1-10)", 1, 10, 5)
ocd = st.sidebar.slider("OCD Level (1-10)", 1, 10, 5)
anxiety = st.sidebar.slider("Anxiety Level (1-10)", 1, 10, 5)
bpm = st.sidebar.slider("BPM", 60, 200, 120)
frequency_pop = st.sidebar.selectbox("Frequency of Pop Music", ["Never", "Rarely", "Sometimes", "Frequently", "Very frequently"])
frequency_rock = st.sidebar.selectbox("Frequency of Rock Music", ["Never", "Rarely", "Sometimes", "Frequently", "Very frequently"])
frequency_metal = st.sidebar.selectbox("Frequency of Metal Music", ["Never", "Rarely", "Sometimes", "Frequently", "Very frequently"])
insomnia = st.sidebar.slider("Insomnia Level (1-10)", 1, 10, 5)

# Checking the shape of the input data
input_data = np.array([[age, hours_per_day, depression, ocd, anxiety, bpm]])
print("Shape of input data:", input_data.shape)

# Encode the frequency options as numerical values
frequency_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Frequently": 3,
    "Very frequently": 4
}

frequency_pop_encoded = frequency_mapping[frequency_pop]
frequency_rock_encoded = frequency_mapping[frequency_rock]
frequency_metal_encoded = frequency_mapping[frequency_metal]

# Make predictions using the loaded models
insomnia_predictions = []
anxiety_predictions = []

# Verify the number of features in the dataset
num_features = 6  # Number of features for each ensemble

# Check the shape of the input array
input_array_shape = np.array([[age, hours_per_day, depression, ocd, anxiety, bpm]]).shape
if input_array_shape[1] != num_features:
    st.error("Error: Input array should have " + str(num_features) + " features, but found " + str(input_array_shape[1]) + " features.")


for model in model_insomnia:
    insomnia_predictions.append(model.predict(np.array([[age, hours_per_day, depression, ocd, anxiety, bpm]]))[0])

for model in model_anxiety:
    anxiety_predictions.append(model.predict(np.array([[ocd, insomnia, depression, frequency_pop_encoded, frequency_rock_encoded, frequency_metal_encoded]]))[0])


# Combine the predictions
ensemble_insomnia_prediction = np.mean(insomnia_predictions)
ensemble_anxiety_prediction = np.mean(anxiety_predictions)


# Display the predicted insomnia and anxiety levels
st.header("Predicted levels of Insomnia and Anxiety")
st.write("Here is what the model has predicted based on your user inputs: ")
st.markdown("<b>Predicted Insomnia Level: </b>"+ str(ensemble_insomnia_prediction) + ".", unsafe_allow_html=True)
st.markdown("<b>Predicted Anxiety Level: </b>"+ str(ensemble_anxiety_prediction) + ".", unsafe_allow_html=True)


# Display recommendations based on predictions
st.header("Music recommendations for you:")
if (ensemble_insomnia_prediction > 5):
    st.write("Your insomnia level is high. Here are some music that can help you fall asleep easier:")
    st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1DX2FOpzhO8XSW?utm_source=generator", height=352)
if (ensemble_anxiety_prediction > 5):
     st.write("Your anxiety level is high. Here are some music that can help calm down your anxiety :")
     st.components.v1.iframe("https://open.spotify.com/embed/playlist/37i9dQZF1EIg42NGihn0NZ?utm_source=generator", height=352)

st.header("Resources")
st.markdown(
"""
Music & Mental Health Survey Results Dataset
- Link: https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results. 
- The dataset is available for access on Kaggle. Kaggle is the world's largest data science community, a subsidiary of Google. 
- This dataset is CC0 Licensed on Public Domain and has a usability rating of 10.0. 
"""
)
st.markdown(
"""
scikit-learn 1.3.1: Python Machine Learning Library
- Link: https://scikit-learn.org/stable/index.html.
- Library used to build machine learning models.
- Models used: sklearn.ensemble.RandomForestRegressor, sklearn.ensemble.GradientBoostingClassifier, Support Vector Machine, Neural Networks.
"""
)
st.markdown(
"""
NumPy 1.26.0: Package for scientific computing with Python
- Link: https://numpy.org/
- Library used to conduct statistical analysis.
"""
)

st.header("About author")
st.write("Hi everyone, my name's Chi. My passion is using technology in ways that prioritize human rights and wellness as the foundation for technological development.")
st.write("My interests encompass a wide array of scientific disciplines, from data, machine learning, artificial intelligence, statistics, math, physics, and chemistry, to space, social science, political science, biology, and materials science.")
st.write("These topics provide the scientific basis for the works in which I actively work to create values in favor of social goods and the advancement of human rights.")
st.write("Through my endeavors, I seek to harness technology and domain expertise to create positive impacts in the world.")
st.write("In the process, I am eager to collaborate with like-minded individuals who share this vision and are dedicated to making a difference.")
st.write("Through collaboration, we can work towards a future where technology serves as a tool for empowerment and scientific inquiry drives positive change, ultimately leading to a world that prioritizes human rights, well-being, progress and access for everyone")
st.markdown(
"""
If you wanna collaborate or just share and discuss stuff together, here are my contacts and socials: 
- Email: roselletopia@gmail.com
- Tik tok: @cheesecake9217
"""
)
