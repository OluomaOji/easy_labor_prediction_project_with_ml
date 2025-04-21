# Standard libraries
import os
import sys
import pandas as pd
import numpy as np
import joblib  # ✅ Needed for saving label encoders and scaler

# Scikit-learn and Imbalanced-learn modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Custom modules (presumably part of your project)
from src.utils.config import FeatureEngineeringConfig
from src.logging import get_logger
from src.exception import EasyLaborPredictionException

# Initialize logger for tracking
logging = get_logger(__name__)

class FeatureEngineering:
    def __init__(self):
        # Load configuration for paths and settings
        self.feature_engineering_config = FeatureEngineeringConfig()

    def initiate_feature_engineering(self):
        """
        Main method to carry out feature engineering pipeline:
        Steps:
        1) Load the dataset
        2) Encode categorical features (Label Encoding)
        3) Normalize numerical features
        4) Encode target
        5) Apply SMOTE for balancing
        6) Save full balanced dataset
        7) Split into train and test sets
        8) Save train and test data
        9) Save Label Encoders and Scaler for future use
        """
        try:
            logging.info("Starting the Feature Engineering Process....")
            
            # 1) Load the dataset
            df = pd.read_csv(self.feature_engineering_config.data_path)
            df = df.drop(columns=['case_id'])
            logging.info(f"Data Loaded from: {self.feature_engineering_config.data_path}")

            # Define the target column
            target_col = 'case_status'

            # 2) Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Define categorical columns
            categorical_columns = [
                'continent', 'education_of_employee', 'has_job_experience', 
                'requires_job_training', 'region_of_employment', 'full_time_position', 'unit_of_wage'
            ]

            # ✅ Apply Label Encoding using a separate encoder per column
            label_encoders = {}
            for column in categorical_columns:
                le = LabelEncoder()
                X[column] = le.fit_transform(X[column])
                label_encoders[column] = le

            # ✅ Save the dictionary of label encoders
            joblib.dump(label_encoders, self.feature_engineering_config.label_encoder_pkl)
            logging.info("Label encoders saved as label_encoders.pkl")

            # 3) Normalize numerical columns using MinMaxScaler
            numeric_columns = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']
            scaler = MinMaxScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

            # ✅ Save the fitted scaler
            joblib.dump(scaler, self.feature_engineering_config.scaler_pkl)
            logging.info("Scaler saved as scaler.pkl")

            # 4) Encode the target variable
            target_encoder = LabelEncoder()
            y_encoded = target_encoder.fit_transform(y)

            # ✅ Save target encoder too (optional, but useful for decoding later)
            joblib.dump(target_encoder, self.feature_engineering_config.target_encoder_pkl)
            logging.info("Target encoder saved as target_encoder.pkl")

            # 5) Apply SMOTE to the entire dataset
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

            # 6) Save full resampled dataset
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df[target_col] = y_resampled
            resampled_df.to_csv(self.feature_engineering_config.output_path, index=False)
            logging.info(f"Resampled dataset saved to {self.feature_engineering_config.output_path}")
            logging.info(f"Resampled class distribution:\n{resampled_df[target_col].value_counts()}")

            # 7) Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            # 8) Save train and test datasets
            train_resampled = pd.DataFrame(X_train, columns=X.columns)
            train_resampled[target_col] = y_train
            test_data = pd.DataFrame(X_test, columns=X.columns)
            test_data[target_col] = y_test

            train_resampled.to_csv(self.feature_engineering_config.train_path, index=False)
            test_data.to_csv(self.feature_engineering_config.test_path, index=False)
            logging.info(f"Train dataset saved to {self.feature_engineering_config.train_path}")
            logging.info(f"Test dataset saved to {self.feature_engineering_config.test_path}")

            return self.feature_engineering_config.train_path, self.feature_engineering_config.test_path

        except Exception as e:
            raise EasyLaborPredictionException(message=str(e), error=sys.exc_info())

# Run this process if the script is executed directly
if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering_processed = feature_engineering.initiate_feature_engineering()
    logging.info("Feature Engineering Process Completed...")
