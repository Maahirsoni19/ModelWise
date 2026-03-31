import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
import shap
from model_engine import (
    detect_task_type, train_and_evaluate,
    get_feature_importance, save_best_model,
    tune_best_model, prepare_data
)
plt.style.use('dark_background')

st.set_page_config(page_title="ModelWise", page_icon="🤖", layout="wide")
st.title("ModelWise: Instant ML Model Selector 🤖")
st.caption("Upload a CSV → Get the best model, metrics, and feature importance automatically.")

# ── Upload ──
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head(), use_container_width=True)

    target_col = st.selectbox("Select your target column (what you want to predict)", df.columns)

    if target_col:
        task = detect_task_type(df, target_col)
        if task == 'classification':
            st.info("🔵 Task detected: **Classification** — predicting categories")
        else:
            st.info("🟠 Task detected: **Regression** — predicting a number")

        if st.button("🚀 Train Models"):
            @st.cache_data
            def cached_train(df, target_col):
                return train_and_evaluate(df, target_col)

            with st.spinner("Training models... (first run takes longer on cloud)"):
                results, task, feature_names = cached_train(df, target_col)
            # ── Save to session state ──
            st.session_state['results']       = results
            st.session_state['task']          = task
            st.session_state['feature_names'] = feature_names
            st.session_state['df']            = df
            st.session_state['target_col']    = target_col
            st.session_state['tuned']         = None  # reset tuning

        # ── Show results if they exist in session ──
        if 'results' in st.session_state:
            results       = st.session_state['results']
            task          = st.session_state['task']
            feature_names = st.session_state['feature_names']
            df            = st.session_state['df']
            target_col    = st.session_state['target_col']

            # Dropped columns warning
            dropped = [col for col in df.columns if df[col].nunique() == len(df)]
            if dropped:
                st.warning(f"⚠️ Auto-dropped useless columns: {dropped}")

            st.subheader("📊 Model Comparison (Best → Worst)")
            display_df = pd.DataFrame([
                {k: v for k, v in r.items() if not k.startswith('_')}
                for r in results
            ])
            st.dataframe(display_df, use_container_width=True)

            best       = results[0]
            best_model = best['_model_obj']
            st.success(f"🏆 Best Model: **{best['Model']}** — CV Score: {best['CV Score']}")

            # ── SHAP Explainability ──
            st.subheader("🔍 SHAP Explainability")
            from model_engine import get_shap_values
            X, y = prepare_data(df, target_col)

            explainer, shap_values = get_shap_values(best_model, X, best['Model'])

            if shap_values is not None:
                # Handle binary classification (RF returns list of arrays)
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values
                base_val = explainer.expected_value
                if isinstance(base_val, list):
                    base_val = base_val[1]

                # Summary bar plot
                st.write("**Which features matter most overall?**")
                #col1, col2 = st.columns([2, 1])
                #with col1:
                fig, ax = plt.subplots(figsize=(12, 6))
                shap.summary_plot(sv, X, plot_type="bar", show=False, max_display=15)
                plt.subplots_adjust(left=0.2)  # ← gives space for long names
                st.pyplot(fig)
                plt.clf()

                # Waterfall for row 0
                st.write("**Why did row #1 get this prediction?**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 3))
                        
                        # Handle all SHAP array shapes
                        if isinstance(sv, list):
                            # multiclass list format
                            row_shap = sv[1][0]
                            base_val_row = base_val[1] if hasattr(base_val, '__len__') else base_val
                        elif sv.ndim == 3:
                            # 3D array (rows, features, classes)
                            row_shap = sv[0, :, 1]
                            base_val_row = base_val[1] if hasattr(base_val, '__len__') else base_val
                        else:
                            # 2D array (binary classification or regression)
                            row_shap = sv[0]
                            base_val_row = base_val

                        shap.waterfall_plot(
                            shap.Explanation(
                                values=row_shap,
                                base_values=float(base_val_row),
                                data=X.iloc[0],
                                feature_names=feature_names
                            ),
                            show=False
                        )
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.info(f"Waterfall plot unavailable: {e}")

            else:
                # Fallback to basic feature importance
                fi = get_feature_importance(best_model, feature_names)
                if fi is not None:
                    st.subheader("🔍 Feature Importance")
                    st.bar_chart(fi.set_index('Feature')['Importance'])

            # ── Confusion Matrix ──
            if task == 'classification':
                st.subheader("🔢 Confusion Matrix")
                cm = best['_confusion_matrix']
                col1, col2 = st.columns([1, 2])
                with col1:
                    fig, ax = plt.subplots(figsize=(5, 4), facecolor='#0e1117')
                    ax.set_facecolor('#0e1117')
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                annot_kws={'color': 'white'})
                    ax.tick_params(colors='white')
                    ax.xaxis.label.set_color('white')
                    ax.yaxis.label.set_color('white')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    plt.tight_layout()
                    st.pyplot(fig)

            # ── Save Model ──
            path = save_best_model(best_model)
            st.success(f"💾 Model saved to `{path}` — ready for API use!")

            # Download button
            with open(path, 'rb') as f:
                st.download_button(
                    label="⬇️ Download Best Model (.pkl)",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream"
                )

            # ── Tune Button ──
            if st.button("⚡ Tune Best Model"):
                with st.spinner(f"Tuning {best['Model']}... (may take 1-2 minutes)"):
                    X, y = prepare_data(df, target_col)
                    tuned_model, best_params, tuned_score = tune_best_model(
                        best_model,
                        best['Model'],
                        X, y, task
                    )
                st.session_state['tuned'] = {
                    'model':       tuned_model,
                    'params':      best_params,
                    'score':       tuned_score,
                    'base_score':  best['CV Score']
                }

            # ── Show Tuning Results if done ──
            if st.session_state.get('tuned'):
                tuned = st.session_state['tuned']
                st.subheader("⚡ Tuning Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Before Tuning", tuned['base_score'])
                with col2:
                    if tuned['score'] is not None:
                        st.metric("After Tuning", tuned['score'],
                                delta=round(tuned['score'] - tuned['base_score'], 4))
                    else:
                        st.info("This model has no hyperparameters to tune.")
                st.write("Best Parameters Found:", tuned['params'])
                path = save_best_model(tuned['model'], 'models/tuned_model.pkl')
                st.success(f"💾 Tuned model saved to `{path}`")

                with open(path, 'rb') as f:
                    st.download_button(
                        label="⬇️ Download Tuned Model (.pkl)",
                        data=f,
                        file_name="tuned_model.pkl",
                        mime="application/octet-stream"
                    )

            # ── Live Prediction Interface ──
                st.subheader("🔮 Make a Prediction")
                st.write("Fill in the values below to get a prediction from the best model.")

                # Use tuned model if available, otherwise best model
                active_model = st.session_state['tuned']['model'] if st.session_state.get('tuned') else best_model

                # Get the prepared feature names
                # Use original df for display, not encoded version
                X_sample = df.drop(columns=[target_col])
                dropped_cols = [col for col in df.columns if df[col].nunique() == len(df)]
                X_sample = X_sample.drop(columns=dropped_cols, errors='ignore')

                input_data = {}
                cols = st.columns(3)  # 3 inputs per row

                for i, col_name in enumerate(X_sample.columns):
                    with cols[i % 3]:
                        col_data = X_sample[col_name]
                        # If column has few unique values → selectbox
                        if col_data.nunique() <= 10:
                            options = sorted(col_data.unique().tolist(), key=str)
                            input_data[col_name] = st.selectbox(col_name, options, key=f"pred_{col_name}")
                        else:
                            # Numeric input with min/max from data
                            min_val = float(col_data.min())
                            max_val = float(col_data.max())
                            mean_val = float(col_data.mean())
                            input_data[col_name] = st.number_input(
                                col_name,
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"pred_{col_name}"
                            )

                if st.button("🔮 Predict"):
                    input_df = pd.DataFrame([input_data])
                    
                    # Encode categorical columns
                    for col in input_df.select_dtypes(include='object').columns:
                        input_df[col] = input_df[col].astype('category').cat.codes
                    
                    # Fill any missing
                    input_df = input_df.fillna(0)
                    
                    try:
                        prediction = active_model.predict(input_df)[0]
                        if task == 'classification' and hasattr(active_model, 'predict_proba'):
                            proba = active_model.predict_proba(input_df)[0]
                            confidence = round(max(proba) * 100, 1)
                            st.success(f"**Prediction: {prediction}** — Confidence: {confidence}%")
                        else:
                            st.success(f"**Predicted Value: {round(float(prediction), 2)}**")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")