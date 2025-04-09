import logging
import os


# set the directory for the logs
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

# set the directory 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir,"easylaborprediction.logs")),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    return logging.getLogger(name)