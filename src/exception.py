import sys
import traceback
from src.logging import get_logger

logging = get_logger(__name__)

class EasyLaborPredictionException(Exception):
    def __init__(self,message=None,error=None):
       super().__init__(message)
       self.error = error

       if message:
           logging.error(f"Error Message: {message}")
       if error:
            logging.error(f"Error Details: {error}"+ traceback.format_exc())