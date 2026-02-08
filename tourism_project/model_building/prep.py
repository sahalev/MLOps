# Data manipulation
import pandas as pd
import os

# Train-test split
from sklearn.model_selection import train_test_split

# Hugging Face API
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset directly from Hugging Face
DATASET_PATH = "hf://datasets/sahalev/tourism-customer-purchase/tourism.csv"
dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# -----------------------------
# Drop identifier column
# -----------------------------
dataset.drop(columns=["CustomerID"], inplace=True)

# -----------------------------
# Define target variable
# -----------------------------
target = "ProdTaken"

# -----------------------------
# Numerical features
# -----------------------------
numeric_features = [
    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"
]

# -----------------------------
# Categorical features
# -----------------------------
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]

# -----------------------------
# Feature matrix and target
# -----------------------------
X = dataset[numeric_features + categorical_features]
y = dataset[target]

# -----------------------------
# Train-test split
# -----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Save splits locally
# -----------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test split completed and saved.")

# -----------------------------
# Upload splits to Hugging Face Dataset
# -----------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="sahalev/tourism-customer-purchase",
        repo_type="dataset",
    )

print("Processed datasets uploaded to Hugging Face.")
