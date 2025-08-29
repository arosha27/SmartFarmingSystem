import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from Model_def import model


################ Load Data ################################
df = pd.read_csv("Data/processed/processed_farming_data.csv")



################ Separating features and target ############
X = df.drop(columns = ["crop_disease_risk" , "crop_disease_risk_code"] , axis=1)
y = df["crop_disease_risk_code"]



####################### Train_Test_Split#####################
X_train , X_temp , y_train, y_temp = train_test_split(X,y , test_size = 0.2 , random_state = 42 , stratify = y)
X_test , X_val , y_test , y_val = train_test_split(X_temp,y_temp , test_size = 0.5 , random_state = 42 , stratify = y_temp)



#################### Standardization ##################
scaler = StandardScaler()

X_train_scaled =scaler.fit_transform(X_train) 
X_test_scaled =scaler.transform(X_test) 
X_val_scaled =scaler.transform(X_val) 



###################### Model Training #################
# model is a dictionary defined in Model_def.py .
# Thus looping over this dictionary and training the models one by one 
def train_and_evaluate(X_train_scaled, X_val_scaled, X_test_scaled,
                       y_train, y_val, y_test):

    for name, md in model.items():
        print("="*60)
        print(f"Training {name.upper()} ...")

        if name == "xgb":
            # Train with validation monitoring
            md.fit(
                X_train_scaled, y_train,
                eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                # eval_metric="mlogloss",
                verbose=False
            )

            # Extract log loss values
            results = md.evals_result()
            train_loss = results['validation_0']['mlogloss']
            val_loss = results['validation_1']['mlogloss']

            # Plot loss curves
            plt.figure(figsize=(6,4))
            plt.plot(train_loss, label="Train LogLoss")
            plt.plot(val_loss, label="Val LogLoss")
            plt.title("XGBoost Training vs Validation Loss")
            plt.xlabel("Boosting Rounds")
            plt.ylabel("LogLoss")
            plt.legend()
            plt.show()

            # Evaluate on validation
            y_val_pred = md.predict(X_val_scaled)
            print(f"XGB Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")

        else:
            # Standard sklearn models
            md.fit(X_train_scaled, y_train)

        # Evaluate on test set (all models)
        y_test_pred = md.predict(X_test_scaled)
        print(f"{name.upper()} Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
        print(classification_report(y_test, y_test_pred))
        print("="*60, "\n")
        
        
        
        

############### Calling the Training & Evaluating Function #################
train_and_evaluate(X_train_scaled, X_val_scaled, X_test_scaled,
                       y_train, y_val, y_test)


############### Saving the best Model (XGBOOST) ##############################
import pickle

xgb_model = model["xgb"]

with open("../API/models/xgboost_classifier.pickle" , "wb") as f:
    pickle.dump(xgb_model , f)
    
with open("../API/models/scaler.pickle", "wb") as f:
    pickle.dump(scaler, f)