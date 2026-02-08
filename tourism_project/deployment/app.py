import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# ----------------------------------
# Load model from Hugging Face Hub
# ----------------------------------
model_path = hf_hub_download(
    repo_id="sahalev/tourism-purchase-model",
    filename="best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("Wellness Tourism Package Purchase Prediction")
st.write(
    """
    This application predicts whether a customer is likely to purchase the
    **Wellness Tourism Package** based on their demographic details and
    interaction history.
    """
)

st.subheader("Enter Customer Details")

# ----------------------------------
# User inputs
# ----------------------------------
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips per Year", min_value=0, value=2)
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, value=50000.0)
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
ProductPitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe"]
)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=2)
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=20)

# ----------------------------------
# Prepare input dataframe
# ----------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "PreferredPropertyStar": PreferredPropertyStar,
    "NumberOfTrips": NumberOfTrips,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "NumberOfFollowups": NumberOfFollowups,
    "DurationOfPitch": DurationOfPitch,
    "TypeofContact": TypeofContact,
    "Occupation": Occupation,
    "Gender": Gender,
    "MaritalStatus": MaritalStatus,
    "ProductPitched": ProductPitched,
    "Designation": Designation
}])

# ----------------------------------
# Prediction
# ----------------------------------
classification_threshold = 0.5

if st.button("Predict Purchase"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = int(prediction_proba >= classification_threshold)

    if prediction == 1:
        st.success(
            f" The customer is **likely to purchase** the Wellness Tourism Package.\n\n"
            f"Purchase Probability: **{prediction_proba:.2f}**"
        )
    else:
        st.warning(
            f" The customer is **unlikely to purchase** the Wellness Tourism Package.\n\n"
            f"Purchase Probability: **{prediction_proba:.2f}**"
        )
