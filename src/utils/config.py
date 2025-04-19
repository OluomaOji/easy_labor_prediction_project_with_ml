import sys
import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    save_dir: str=os.path.join('artifacts') # Directory where artifacts (e.g the csv files) will be saved
    raw_data_path: str=os.path.join('data','raw','olfc_data.pdf') # Path to the dataset in pdf format
    data_path: str=os.path.join('artifacts','dataset','olfc_data.csv') # Path where the raw data will be saved in csv format.
    os.makedirs(save_dir,exist_ok=True) # Create the saved directory if it doesn't exist

@dataclass
class EDAConfig:
    data_path: str=os.path.join('artifacts','dataset','olfc_data.csv')
    out_dir: str=os.path.join('artifacts','eda','eda_report')

@dataclass
class FeatureEngineeringConfig:
    data_path: str=os.path.join('artifacts','dataset','olfc_data.csv')
    output_path: str=os.path.join('artifacts','feature_eng','processed_data.csv')
    train_path: str=os.path.join('artifacts','feature_eng','train_data.csv')
    test_path: str=os.path.join('artifacts','feature_eng','test_data.csv')

@dataclass
class ModelTrainingConfig:
    MLFLOW_TRACKING_URI: str= 'https://dagshub.com/OluomaOji/easy_labor_prediction_project_with_ml.mlflow' # MLflow tracking server URI.
    #MLFLOW_TRACKING_USERNAME: str = os.environ.get("MLFLOW_TRACKING_USERNAME")# MLflow tracking username.
    #MLFLOW_TRACKING_PASSWORD: str = os.environ.get("MLFLOW_TRACKING_PASSWORD")  # MLflow tracking password.
    MLFLOW_TRACKING_USERNAME: str = 'OluomaOji'
    MLFLOW_TRACKING_PASSWORD: str = "760c90844a305272f1981be1879bceef551c3385"
    train_path: str=os.path.join('artifacts','feature_eng','train_data.csv')# Path to the training data file.
    test_path: str=os.path.join('artifacts','feature_eng','test_data.csv') # Path to the testing data file.
    best_model: str=os.path.join('artifacts','model.pkl') # Path to save the best model after training.

@dataclass
class HyperParameterTuningConfig:
    MLFLOW_TRACKING_URI: str= 'https://dagshub.com/OluomaOji/easy_labor_prediction_project_with_ml.mlflow' # MLflow tracking server URI.
    #MLFLOW_TRACKING_USERNAME: str = os.environ.get("MLFLOW_TRACKING_USERNAME")# MLflow tracking username.
    #MLFLOW_TRACKING_PASSWORD: str = os.environ.get("MLFLOW_TRACKING_PASSWORD")  # MLflow tracking password.
    MLFLOW_TRACKING_USERNAME: str = 'OluomaOji'
    MLFLOW_TRACKING_PASSWORD: str = "760c90844a305272f1981be1879bceef551c3385"
    train_path: str=os.path.join('artifacts','feature_eng','train_data.csv')# Path to the training data file.
    test_path: str=os.path.join('artifacts','feature_eng','test_data.csv') # Path to the testing data file.
    best_model: str=os.path.join('artifacts','model.pkl') # Path to save the best model after training.
