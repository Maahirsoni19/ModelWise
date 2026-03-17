import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model_engine import (
    detect_task_type, train_and_evaluate,
    get_feature_importance, save_best_model,
    tune_best_model, prepare_data
)

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
            with st.spinner("Training 4 models... (usually < 30 seconds)"):
                results, task, feature_names = train_and_evaluate(df, target_col)

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

            # ── Feature Importance ──
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
                    fig, ax = plt.subplots(figsize=(3, 2.5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
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
                    st.metric("After Tuning", tuned['score'],
                              delta=round(tuned['score'] - tuned['base_score'], 4))
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