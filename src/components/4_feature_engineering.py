# Standard libraries
import os
import sys
import pandas as pd
import numpy as np

# Scikit-learn and Imbalanced-learn modules
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import FunctionTransformer

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
        2) Encode categorical features
        3) Normalize numerical features
        4) Encode target
        5) Apply SMOTE for balancing
        6) Save full balanced dataset
        7) Split into train and test sets
        8) Save train and test data
        """
        try:
            # 1) Load the dataset
            logging.info("Starting the Feature Engineering Process....")
            df = pd.read_csv(self.feature_engineering_config.data_path)
            logging.info(f"Data Loaded from: {self.feature_engineering_config.data_path}")

            # Define the target column
            target_col = 'case_status'

            # 2) Separate features and target
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Define categorical columns
            categorical_columns = [
                'case_id','continent', 'education_of_employee', 'has_job_experience', 
                'requires_job_training', 'region_of_employment', 'full_time_position','unit_of_wage'
            ]

            # Apply Label Encoding to all categorical columns
            label_encoder = LabelEncoder()
            for column in categorical_columns:
                X[column] = label_encoder.fit_transform(X[column])

            # Define numeric columns
            numeric_columns = ['no_of_employees', 'yr_of_estab', 'prevailing_wage']

            # Normalize numerical columns using MinMaxScaler
            scaler = MinMaxScaler()
            X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

            # Encode the target variable
            y_encoded = label_encoder.fit_transform(y)

            # 3) Apply SMOTE to the entire dataset before train-test split
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_encoded)

            # 4) Log class distribution after resampling
            resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_df[target_col] = y_resampled
            logging.info(f"Value counts of the resampled target variable (case_status):\n{resampled_df[target_col].value_counts()}")

            # 5) Save the entire resampled dataset
            resampled_df.to_csv(self.feature_engineering_config.output_path, index=False)
            logging.info(f"Full resampled dataset saved to {self.feature_engineering_config.output_path}")

            # 6) Now perform train-test split on the resampled data
            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            # 7) Rebuild train/test DataFrames for saving
            train_resampled = pd.DataFrame(X_train, columns=X.columns)
            train_resampled[target_col] = y_train

            test_data = pd.DataFrame(X_test, columns=X.columns)
            test_data[target_col] = y_test

            # 8) Save train and test datasets
            train_resampled.to_csv(self.feature_engineering_config.train_path, index=False)
            test_data.to_csv(self.feature_engineering_config.test_path, index=False)

            logging.info(f"Train dataset saved to {self.feature_engineering_config.train_path}")
            logging.info(f"Test dataset saved to {self.feature_engineering_config.test_path}")

            return self.feature_engineering_config.train_path, self.feature_engineering_config.test_path

        except Exception as e:
            # Custom exception handling with logging
            raise EasyLaborPredictionException(message=str(e), error=sys.exc_info())

# Run this process if the script is executed directly
if __name__ == "__main__":
    feature_engineering = FeatureEngineering()
    feature_engineering_processed = feature_engineering.initiate_feature_engineering()
    logging.info("Feature Engineering Process Completed...")
