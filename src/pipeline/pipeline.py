import os
import sys

from src.exception import EasyLaborPredictionException
from src.logging import get_logger

from src.components.data_ingestion import DataIngestion
from src.components.EDA import EDA
from src.components.feature_engineering import FeatureEngineering
from src.components.model_training import ModelTraining
from src.components.hyperparameter_tuning import HyperparameterTuning

logging = get_logger(__name__)

class RunPipeline:
    def __init__(self):
        pass

    def initiate_pipeline(self):
        try:
           # 1) Data Ingestion
           logging.info("Starting the Data Ingestion Process")
           ingestion = DataIngestion()
           ingestion.initiate_data_ingestion()
           logging.info("Data Ingestion Completed")

           # 2) EDA
           logging.info("Starting the Explorative Data Analysis")
           eda = EDA()
           eda.initialising_eda()
           logging.info('EDA Process Completed')

           # 3) Feature Engineering
           logging.info("Starting the Feature Engineering Process")
           feature_engineering = FeatureEngineering()
           feature_engineering.initiate_feature_engineering()
           logging.info("Feature Engineering Completed")

           # 4) Model Training
           logging.info("Starting the Model Training Process")
           model_training = ModelTraining()
           model_training.initiate_model_training()
           logging.info("Model Training Completed")

           # 5) Hyperparameter Tuning
           logging.info("Starting the Hyperparameter Tuning")
           xgboost_tuning = HyperparameterTuning()
           xgboost_tuning.initiate_hyperparameter_tuning()
           logging.info("Hyperparameter Tuning Completed")


        except Exception as e:
            raise EasyLaborPredictionException(message=str(e),error=sys.exc_info())
        
if __name__ == "__main__":
    pipeline = RunPipeline()
    pipeline.initiate_pipeline()