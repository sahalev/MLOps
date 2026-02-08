# -----------------------------
# Data & preprocessing
# -----------------------------
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# -----------------------------
# Model training & evaluation
# -----------------------------
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# -----------------------------
# Utilities
# -----------------------------
import joblib
import os
import mlflow

# -----------------------------
# Hugging Face
# -----------------------------
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-mlops-training")

# -----------------------------
# Hugging Face API
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))

# -----------------------------
# Load train/test data from HF
# -----------------------------
BASE_DATASET = "hf://datasets/sahalev/tourism-customer-purchase"

Xtrain = pd.read_csv(f"{BASE_DATASET}/Xtrain.csv")
Xtest  = pd.read_csv(f"{BASE_DATASET}/Xtest.csv")
ytrain = pd.read_csv(f"{BASE_DATASET}/ytrain.csv").values.ravel()
ytest  = pd.read_csv(f"{BASE_DATASET}/ytest.csv").values.ravel()

print("Training and testing data loaded.")

# -----------------------------
# Feature definitions
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

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "ProductPitched",
    "Designation"
]

# -----------------------------
# Handle class imbalance
# -----------------------------
class_weight = ytrain.tolist().count(0) / ytrain.tolist().count(1)

# -----------------------------
# Preprocessing pipeline
# -----------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -----------------------------
# XGBoost model
# -----------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# -----------------------------
# Hyperparameter grid
# -----------------------------
param_grid = {
    "xgbclassifier__n_estimators": [100, 150],
    "xgbclassifier__max_depth": [3, 4],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.5, 0.7]
}

# -----------------------------
# Pipeline
# -----------------------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# -----------------------------
# Training with MLflow
# -----------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        n_jobs=-1,
        scoring="f1"
    )

    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # -----------------------------
    # Save model
    # -----------------------------
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    print("Best model saved locally and logged to MLflow.")

# -----------------------------
# Upload model to Hugging Face
# -----------------------------
repo_id = "sahalev/tourism-purchase-model"
repo_type = "model"

try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print("Model repo exists. Uploading model...")
except RepositoryNotFoundError:
    print("Creating model repo...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Model uploaded to Hugging Face Model Hub.")
