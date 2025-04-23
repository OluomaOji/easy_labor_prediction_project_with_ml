# import libraries
import os
import sys

import pandas as pd
import numpy as np
from io import StringIO

# Custom imports
from src.utils.config import EDAConfig
from src.logging import get_logger
from src.exception import EasyLaborPredictionException

# initialise the logger
logging = get_logger(__name__)

class EDA:
    def __init__(self):
        # loading configuration
        self.EDA_config=EDAConfig()

    def initialising_eda(self):
        """
        Initialising EDA process
        1. Loading the Dataset from the EDA data_path
        2. Generating the EDA Summary
        3. Saving the EDA Summary as pdf into the output_dir

        """
        try:
            #1. Loading the Dataset from the EDA data_path
            logging.info("EDA processing initialising......")
            logging.info(f"Loading data from: {self.EDA_config.data_path}")
            df = pd.read_csv(self.EDA_config.data_path)
            logging.info(f"Data loaded successfully loaded")

            #2. Generating the EDA Summary 
            #3. Saving the EDA Summary as pdf into the output_dir

            # Capturing info() output to string
            info_str = StringIO()
            df.info(buf=info_str)
            info = info_str.getvalue()

            logging.info("Generating the EDA Summary")
            logging.info(f"Data Head: {df.head()}")
            logging.info(f"Data Shape: {df.shape}")
            logging.info(f"Data Info: {info}")
            logging.info(f"Data Describe: {df.describe()}")
            logging.info(f"Duplicated Values: {df.duplicated().sum()}")
            logging.info(f"Missing Values: \n{df.isnull().sum()}")
            logging.info(f"Unique Values: \n{df.nunique()}")

            eda_path = self.EDA_config.out_dir
            with open(eda_path,'w') as f:
                f.write(f"Data Head: {df.head()}\n")
                f.write(f"Data Shape: \n{df.shape}\n")
                f.write(f"Data Info: \n{info}\n")
                f.write(f"Data Describe: \n{df.describe()}\n")
                f.write(f"Duplicated Values: \n{df.duplicated().sum}\n")
                f.write(f"Missing Values: \n{df.isnull().sum()}\n")
                f.write(f"Unique Values: \n{df.nunique()}\n")
                f.write(f"Number of Unique Categories in Case Status: \n{df['case_status'].value_counts()}")


            logging.info(f"EDA Summary saved to: {eda_path}")

        except Exception as e:
            raise EasyLaborPredictionException(message=str(e),error=sys.exc_info())
        