import os
import sys
import pandas as pd
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

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from joblib import dump

from src.logging import get_logger
from src.exception import EasyLaborPredictionException
from src.utils.config import ModelTrainingConfig

logging = get_logger(__name__)

class ModelTraining:
    def __init__(self):
        self.model_training_config=ModelTrainingConfig()

    def build_lstm_model(self, input_shape):
        """
        Builds an LSTM model for binary classification.
        """
        model = Sequential()
        model.add(LSTM(64, input_shape=(input_shape, 1), return_sequences=True))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def initiate_model_training(self):
        try:
            train_data = pd.read_csv(self.model_training_config.train_path)
            test_data = pd.read_csv(self.model_training_config.test_path)
            logging.info("Training and Test Data Loaded Successfully")

            target_column='case_status'

            X_train=train_data.drop(columns=target_column)
            y_train=train_data[target_column]
            X_test=test_data.drop(columns=target_column)
            y_test=test_data[target_column]

            models ={ 
                "Logistic Regression": LogisticRegression(solver="liblinear",max_iter=1000),
                "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
                "Xgboost": xgb.XGBClassifier(max_depth=4,alpha=10,n_estimators=100),
                "LSTM":self.build_lstm_model(X_train.shape[1])
            }
            
            # Setting the metric for the best model
            best_model=None
            best_model_name=None
            best_roc_auc= float("-inf")

            # Set up MLFLOW for tracking experiments
            os.environ["MLFLOW_TRACKING_URI"] = self.model_training_config.MLFLOW_TRACKING_URI
            os.environ["MLFLOW_TRACKING_USERNAME"] = self.model_training_config.MLFLOW_TRACKING_USERNAME
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self.model_training_config.MLFLOW_TRACKING_PASSWORD

            mlflow_tracking_uri = self.model_training_config.MLFLOW_TRACKING_URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)

            experiment_name = "Easy_Labor_Prediction"
            mlflow.set_experiment(experiment_name)

            # Train and Evaluate Each Model, Log Metrics and Save Artifacts.
            for model_name, model in models.items():
                logging.info(f"\n======= Training Model: {model_name} ======")

                with mlflow.start_run(run_name=model_name):
                    if model_name == "LSTM":
                        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
                        y_pred = model.predict(X_test)
                        y_pred_labels = (y_pred >= 0.5).astype(int)
                        y_proba = y_pred.flatten()
                    else:
                        model.fit(X_train, y_train)
                        y_pred_labels = model.predict(X_test)
                        if hasattr(model, "predict_proba"):
                            y_proba = model.predict_proba(X_test)[:, 1]
                        else:
                            y_proba = model.predict(X_test).flatten()

                    acc = accuracy_score(y_test, y_pred_labels)
                    precision = precision_score(y_test, y_pred_labels)
                    recall = recall_score(y_test, y_pred_labels)
                    f1 = f1_score(y_test, y_pred_labels)
                    roc_auc = roc_auc_score(y_test, y_proba)

                    cm = confusion_matrix(y_test, y_pred_labels)
                    cls_report = classification_report(y_test, y_pred_labels)

                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision",precision)
                    mlflow.log_metric("recall",recall)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("roc_auc", roc_auc)

                    mlflow.log_metric("tn", cm[0,0])
                    mlflow.log_metric("fp", cm[0,1])
                    mlflow.log_metric("fn", cm[1,0])
                    mlflow.log_metric("tp", cm[1,1])

                    logging.info(f"Model: {model_name} | "
                                 f"Accuracy: {acc:.4f} | "
                                 f"Precision: {precision:.4f}"
                                 f"Recall: {recall:.4f}"
                                 f"F1: {f1:.4f} | "
                                 f"ROC-AUC: {roc_auc}")
                    logging.info(f"Confusion Matrix:\n{cm}")
                    logging.info(f"Classification Report:\n{cls_report}")

                    signature = infer_signature(X_train, model.predict(X_train))
                    mlflow.sklearn.log_model(
                        model, 
                        artifact_path="artifacts",
                        signature=signature,
                        input_example=X_train,
                    )

                    mlflow.end_run()

                current_roc = roc_auc if roc_auc else 0.0
                if current_roc > best_roc_auc:
                    best_roc_auc = current_roc
                    best_model = model
                    best_model_name = model_name

            logging.info(f"\nBest Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")

            # Save the best model to .pkl
            if best_model is not None:
                model_path = self.model_training_config.best_model
                dump(best_model, model_path)
                logging.info(f"Best model saved to: {model_path}")

        except Exception as e:
            raise EasyLaborPredictionException(message=str(e),error=sys.exc_info())
        
if __name__=="__main__":
    modeltraining = ModelTraining()
    best_model = modeltraining.initiate_model_training()
    logging.info("Model Training Process Completed...")

