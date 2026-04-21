import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import shap
import pickle

def train_models():
    print("Loading training data...")
    try:
        df = pd.read_csv("training_data.csv")
    except FileNotFoundError:
        print("training_data.csv not found. Please run preprocessing.py first.")
        return

    # Drop identifiers and targets to get features
    X = df.drop(columns=['shipment_id', 'is_delayed', 'delay_days'])
    y_class = df['is_delayed']
    y_reg = df['delay_days']

    # Split for classification
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # Split for regression (only on delayed items, or all if we want zero prediction)
    # Let's train regression on all, it should predict 0 if not delayed
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    print("Training Classification Model (Risk Probability)...")
    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    # Evaluate classification
    preds_c = clf.predict(X_test_c)
    print("Classification Accuracy:", accuracy_score(y_test_c, preds_c))
    print(classification_report(y_test_c, preds_c))

    print("Training Regression Model (Delay Duration)...")
    reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    reg.fit(X_train_r, y_train_r)
    
    # Evaluate regression
    preds_r = reg.predict(X_test_r)
    print("Regression RMSE:", np.sqrt(mean_squared_error(y_test_r, preds_r)))

    print("Generating SHAP Explainer...")
    # We use TreeExplainer for XGBoost
    explainer = shap.TreeExplainer(clf)
    
    print("Saving models...")
    with open('model_clf.pkl', 'wb') as f:
        pickle.dump(clf, f)
    with open('model_reg.pkl', 'wb') as f:
        pickle.dump(reg, f)
    with open('shap_explainer.pkl', 'wb') as f:
        pickle.dump(explainer, f)
        
    # Save feature names for the app
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)

    print("Model training complete.")

if __name__ == "__main__":
    train_models()
