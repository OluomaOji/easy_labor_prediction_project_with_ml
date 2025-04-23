# import libraries
import os
import sys
import pandas as pd

# Custom imports
from src.logging import get_logger
from src.exception import EasyLaborPredictionException
from src.utils.config import DataIngestionConfig

# initialise logging
logging = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        # Loading Configuration
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiating Data Ingestion Process

        1. Read the data file using raw_data_path from the config
        2. Save the dataset as csv in the data_path
        """
        try:
            # 1) Reading the data file
            logging.info("Starting data ingestion process....")
            df=pd.read_csv('data/raw/olfc_data.csv')
            # 2) Saving the data file
            df.to_csv(self.data_ingestion_config.data_path,index=False,header=True)
            logging.info(f"Data saved to CSV at: {self.data_ingestion_config.data_path}")

            return self.data_ingestion_config.data_path
        except Exception as e:
            raise EasyLaborPredictionException(message=str(e),error=sys.exc_info())