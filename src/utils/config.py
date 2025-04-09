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
