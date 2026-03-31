# **ModelWise**
## 🚀 Live Demo
👉 [modelwise.streamlit.app](https://modelwise.streamlit.app)

I built this because I was tired of writing the same model comparison code every time I started a new ML project. You upload a CSV, pick what you want to predict, and it handles the rest.

## **What it does**

Figures out whether your problem is classification or regression, trains four models (LightGBM, XGBoost, Random Forest, and Logistic/Linear Regression), and ranks them using 5-fold cross-validation so the results are actually reliable and not just a lucky train/test split.

It also shows you which features are driving predictions using SHAP, gives you a confusion matrix if you're doing classification, and lets you tune the best model with one click. When you're done, you can download the trained model as a `.pkl` file and plug it straight into an API.

## **Why I made it this way**

Most AutoML tools are either too heavy or too opaque. I wanted something lightweight that I could actually understand and explain in a hackathon presentation. Simple, fast, and honest about what it's doing.

## **Run it locally**
```
pip install -r requirements.txt
streamlit run app.py
```

## **Stack**
- Streamlit
- scikit-learn
- XGBoost
- LightGBM  
- SHAP
- joblib
