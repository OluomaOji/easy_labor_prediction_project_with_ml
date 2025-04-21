import os
import sys
import pandas as pd
import tabula

from src.logging import get_logger
from src.exception import EasyLaborPredictionException
from src.utils.config import DataIngestionConfig

logging = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiating Data Ingestion Process

        1. Read the data file using raw_data_path from the config
        2. Convert the dataset from pdf to csv file
        3. Save the dataset as csv in the data_path
        """
        try:
            
            logging.info("Starting data ingestion process....")
            df=pd.read_csv('data/raw/olfc_data.csv')
            df.to_csv(self.data_ingestion_config.data_path,index=False,header=True)
            logging.info(f"Data saved to CSV at: {self.data_ingestion_config.data_path}")

            return self.data_ingestion_config.data_path
        except Exception as e:
            raise EasyLaborPredictionException(message=str(e),error=sys.exc_info())

if __name__ == "__main__":
    ingestion = DataIngestion()
    csv_path = ingestion.initiate_data_ingestion()
    logging.info("Data Ingestion Completed")