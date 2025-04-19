import pandas as pd
import os
import sys
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from joblib import dump

from src.utils.config import HyperParameterTuningConfig
from src.logging import get_logger
from src.exception import EasyLaborPredictionException

logging = get_logger(__name__)

class HyperparameterTuning:
    def __init__(self):
        self.hyperparameter_config = HyperParameterTuningConfig()

    def initiate_hyperparameter_tuning(self):
        try:
            train_data = pd.read_csv(self.hyperparameter_config.train_path)
            test_data = pd.read_csv(self.hyperparameter_config.test_path)
            logging.info("Training and Test Data Loaded for Hyperparameter Tuning")

            target_column = 'case_status'

            X_train = train_data.drop(columns=[target_column])
            y_train = train_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]

            xgb_clf = XGBClassifier()

            param_dist = {
                'n_estimators': [100, 500, 900, 1100, 1500],
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'gamma': [0, 0.1, 0.2, 0.5],
                'reg_alpha': [0, 0.1, 1],
                'reg_lambda': [1, 1.5, 2]
            }

            model = RandomizedSearchCV(
                xgb_clf,
                param_distributions=param_dist,
                n_iter=20,
                scoring='accuracy',
                cv=3,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            logging.info("Starting hyperparameter tuning...")
            model.fit(X_train, y_train)
            best_model = model.best_estimator_
            best_params = model.best_params_

            # Evaluate the model
            y_proba = best_model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            y_pred = best_model.predict(X_test)
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')  # or 'macro' for multiclass
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')

            # Set up MLFLOW for tracking experiments
            os.environ["MLFLOW_TRACKING_URI"] = self.hyperparameter_config.MLFLOW_TRACKING_URI
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.hyperparameter_config.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.hyperparameter_config.MLFLOW_TRACKING_PASSWORD

            # Set MLflow URI and experiment
            mlflow_tracking_uri = self.hyperparameter_config.MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            experiment_name = "Easy_Labor_Prediction"
            mlflow.set_experiment(experiment_name)

            model_name="Xgboost Hyperparameter"

            with mlflow.start_run(run_name=model_name):
                mlflow.log_params(best_params)
                mlflow.log_metric("roc_auc", roc_auc)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc", roc_auc)
                signature = infer_signature(X_train, model.predict(X_train))
                mlflow.sklearn.log_model(
                        model, 
                        artifact_path="artifacts",
                        signature=signature,
                        input_example=X_train,
                    )

                logging.info(f"ROC-AUC on Test Data: {roc_auc:.4f}")
                logging.info(f"Best Parameters: {best_params}")

            # Save the best model
            dump(best_model, self.hyperparameter_config.best_model)
            logging.info(f"Best tuned XGBoost model saved to: {self.hyperparameter_config.best_model}")

        except Exception as e:
            raise EasyLaborPredictionException(message=str(e), error=sys.exc_info())

if __name__ == "__main__":
    tuner = HyperparameterTuning()
    tuner.initiate_hyperparameter_tuning()
    logging.info("XGBoost Hyperparameter Tuning Completed...")
