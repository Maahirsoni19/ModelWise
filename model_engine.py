#creating the brain of ModelWise
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
import shap
from sklearn.metrics import (
    accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix
import joblib
import os


def drop_useless_columns(df):
    cols_to_drop = []
    for col in df.columns:
        # Drop if unique values = total rows (ID columns like PassengerId, Name)
        if df[col].nunique() == len(df):
            cols_to_drop.append(col)
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df, cols_to_drop
# ─────────────────────────────────────────────
# STEP 1: DETECT TASK TYPE
def detect_task_type(df, target_col):
    target = df[target_col]
    unique_vals = target.nunique()
    dtype = target.dtype

    if dtype == 'object' or dtype == 'bool':
        return 'classification'
    elif unique_vals <= 10:
        return 'classification'
    else:
        return 'regression'


# ─────────────────────────────────────────────
# STEP 2: PREPARE DATA
def prepare_data(df, target_col):
    df, dropped = drop_useless_columns(df)
    df = df.copy()
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category').cat.codes

    X = X.fillna(X.median(numeric_only=True))

    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    return X, y


#---------------------------------------------------------
#STEP 3: Training multiple models and comparing them
def train_and_evaluate(df, target_col):
    task = detect_task_type(df, target_col)
    X, y = prepare_data(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task == 'classification':
        models = {
            'LightGBM':          LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost':           XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0),
            'Random Forest':     RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        }
        cv_scoring='accuracy'
    else:
        models = {
            'LightGBM':        LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'XGBoost':         XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
            'Random Forest':   RandomForestRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression(),
        }
        cv_scoring='r2'
    results=[]
    for name, model in models.items(): 
        #implementing 5-fold cross-validation
        cv_scores=cross_val_score(model, X, y, cv=5, scoring=cv_scoring)
        cv_mean= round(cv_scores.mean(),4)
        cv_std= round(cv_scores.std(),4)


        model.fit(X_train, y_train)
        y_pred=model.predict(X_test)

        if task == 'classification':
            results.append({
                'Model': name,
                'CV Score': cv_mean,
                'CV Std': f'± {cv_std}',
                'Accuracy': round(accuracy_score(y_test,y_pred),4),
                'F1 Score': round(f1_score(y_test,y_pred,average='weighted'),4),
                '_confusion_matrix': confusion_matrix(y_test, y_pred),
                'Task': task,
                '_model_obj':model,
            })
        else:
            results.append({
                'Model':name,
                'CV Score':   cv_mean,
                'CV Std':     f'± {cv_std}',
                'MAE': round(mean_absolute_error(y_test,y_pred),4),
                'RMSE': round(np.sqrt(mean_squared_error(y_test,y_pred)),4),
                'R2 Score': round(r2_score(y_test,y_pred),4),
                'Task': task,
                '_model_obj':model,
            })
    results=sorted(results,key=lambda x: x['CV Score'], reverse =True)

    return results,task, X.columns.tolist()
#-----------------------------------------------------------------------------
#hyperparameter tuning for the best model
def tune_best_model(model, model_name, X, y, task):
    """
    Takes the best model and tunes its hyperparameters.
    Returns tuned model + best params + improved score.
    """
    # Parameter grids for each model type
    param_grids = {
        'Random Forest': {
            'n_estimators':      [100, 200, 300],
            'max_depth':         [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf':  [1, 2, 4],
        },
        'XGBoost': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample':     [0.7, 0.8, 1.0],
        },
        'LightGBM': {
            'n_estimators':  [100, 200, 300],
            'max_depth':     [3, 5, 7, -1],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'num_leaves':    [31, 50, 100],
        },
        'Logistic Regression': {
            'C':       [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver':  ['liblinear', 'saga'],
        },
        'Linear Regression': {}  # no hyperparameters worth tuning
    }

    params = param_grids.get(model_name, {})
    if not params:
        return model, {}, None
    
    scoring = 'accuracy' if task == 'classification' else 'r2'
    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=20,          # try 20 random combinations
        cv=5,               # 5-fold CV
        scoring=scoring,
        random_state=42,
        n_jobs=1           # use all CPU cores → faster
    )
    search.fit(X, y)

    return search.best_estimator_, search.best_params_, round(search.best_score_, 4)


#-----------------------------------------------------------------------------
#STEP 4: Feature importance
def get_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        fi=pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return fi
    return None

#-----------------------------------------------------------------------------
def get_shap_values(model, X, model_name):
    """
    Returns SHAP explainer and values for the model.
    Tree-based models use TreeExplainer (fast).
    Others use LinearExplainer.
    """
    try:
        tree_based = ['Random Forest', 'XGBoost', 'LightGBM']
        if model_name in tree_based:
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            explainer   = shap.LinearExplainer(model, X)
            shap_values = explainer.shap_values(X)
        return explainer, shap_values
    except Exception as e:
        return None, None
#-----------------------------------------------------------------------------
#STEP 5: Save the best model
def save_best_model(model, filename='models/best_model.pkl'):
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, filename)
    return filename

