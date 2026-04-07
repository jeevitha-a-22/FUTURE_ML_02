import streamlit as st
import pickle
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="IT Support Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Artifacts
# -------------------------------
with open('saved_models/tfidf_it.pkl', 'rb') as f:
    tfidf_it = pickle.load(f)
with open('saved_models/model_it_cat.pkl', 'rb') as f:
    model_it = pickle.load(f)
with open('saved_models/le_it.pkl', 'rb') as f:
    le_it = pickle.load(f)

# Load dataset for dashboard
try:
    df = pd.read_csv('data/all_tickets_processed_improved_v3.csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['Ticket_Description', 'Topic_group'])

# Load model metrics for dashboard
try:
    metrics_df_it = pd.read_csv('data/model_metrics_it.xls')
except FileNotFoundError:
    metrics_df_it = pd.DataFrame(columns=['Task','Model','Accuracy','Weighted_F1'])

# -------------------------------
# Tabs: Prediction & Dashboard
# -------------------------------
tab1, tab2 = st.tabs(["Predict Ticket", "Dashboard"])

# -------------------------------
# Tab 1: Single ticket prediction
# -------------------------------
with tab1:
    st.header("🎫 Single IT Ticket Prediction")
    ticket_text = st.text_area("Enter IT ticket description:")

    if st.button("Predict Single Ticket"):
        if ticket_text.strip() == "":
            st.warning("Enter a ticket description!")
        else:
            X_input = tfidf_it.transform([ticket_text])
            pred_label = model_it.predict(X_input)
            pred_class = le_it.inverse_transform(pred_label)[0]
            st.success(f"Predicted IT Category: **{pred_class}**")

            if hasattr(model_it, "predict_proba"):
                probs = model_it.predict_proba(X_input)[0]
                prob_dict = {cl: f"{p:.2%}" for cl, p in zip(le_it.classes_, probs)}
                st.subheader("Prediction Probabilities")
                st.json(prob_dict)

    st.markdown("---")
    st.header("📁 Batch Prediction from CSV")
    uploaded_file = st.file_uploader("Upload CSV with ticket descriptions", type="csv")

    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        
        # Auto-detect text column
        text_column = None
        for col in df_batch.columns:
            if col.lower() in ['ticket_description', 'document', 'description', 'text']:
                text_column = col
                break
        
        if not text_column:
            st.error("No text column detected. CSV must have a ticket description column.")
        else:
            X_batch = tfidf_it.transform(df_batch[text_column])
            preds = model_it.predict(X_batch)
            df_batch['Predicted_IT_Category'] = le_it.inverse_transform(preds)

            st.success(f"✅ Batch predictions done using column: **{text_column}**")
            st.dataframe(df_batch)

            # Download button
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name='it_ticket_predictions.csv',
                mime='text/csv'
            )

# -------------------------------
# Tab 2: Executive Dashboard
# -------------------------------
with tab2:
    st.header("📊 Executive IT Ticket Dashboard")

    # Determine dataset: uploaded CSV or original
    if uploaded_file and 'Predicted_IT_Category' in df_batch.columns:
        df_dashboard = df_batch.copy()
        if 'Topic_group' not in df_dashboard.columns:
            df_dashboard.rename(columns={'Predicted_IT_Category':'Topic_group'}, inplace=True)
    else:
        df_dashboard = df.copy()

    # Safe check
    if 'Topic_group' in df_dashboard.columns and not df_dashboard.empty:
        categories = df_dashboard['Topic_group'].unique().tolist()
    else:
        categories = []

    if categories:
        st.sidebar.header("Filter Tickets by Category")
        selected_categories = st.sidebar.multiselect(
            "Select Categories",
            categories,
            default=categories
        )
        df_filtered = df_dashboard[df_dashboard['Topic_group'].isin(selected_categories)]
    else:
        st.info("No categories available to display.")
        df_filtered = pd.DataFrame()

    # -------------------------------
    # KPI Cards with hover info
    # -------------------------------
    total_tickets = len(df_filtered)
    category_counts = df_filtered['Topic_group'].value_counts() if not df_filtered.empty else pd.Series()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Tickets", total_tickets, help="Total tickets in selected categories")
    kpi2.metric("Top Category", category_counts.idxmax() if not category_counts.empty else "N/A",
                category_counts.max() if not category_counts.empty else 0,
                help="Most frequent IT category")
    kpi3.metric("Hardware Tickets", category_counts.get("Hardware",0), help="Number of hardware tickets")
    kpi4.metric("Access Tickets", category_counts.get("Access",0), help="Number of access-related tickets")

    # -------------------------------
    # Bar chart
    # -------------------------------
    if not df_filtered.empty:
        bar_data = category_counts.reset_index()
        bar_data.columns = ['Category','Count']
        bar_chart = alt.Chart(bar_data).mark_bar().encode(
            x=alt.X('Category', sort='-y'),
            y='Count',
            color='Category',
            tooltip=['Category','Count']
        ).properties(title="IT Tickets by Category")
        st.altair_chart(bar_chart, use_container_width=True)

    # -------------------------------
    # Pie chart
    # -------------------------------
    if not df_filtered.empty:
        pie_chart = px.pie(bar_data, names='Category', values='Count', title="Ticket Distribution")
        st.plotly_chart(pie_chart, use_container_width=True)

    # -------------------------------
    # Model F1 comparison
    # -------------------------------
    if not metrics_df_it.empty:
        st.subheader("Model Comparison — Weighted F1 Score")
        f1_chart = px.bar(metrics_df_it, x='Model', y='Weighted_F1', text='Weighted_F1',
                          color='Model', title="Weighted F1 per Model")
        st.plotly_chart(f1_chart, use_container_width=True)

        # -------------------------------
        # Per-category F1 heatmap
        # -------------------------------
        st.subheader("Per-Category F1 Heatmap")
        plt.figure(figsize=(8,4))
        sns.heatmap(metrics_df_it.set_index('Model')[['Weighted_F1']], annot=True, cmap='RdYlGn', fmt=".2f")
        st.pyplot(plt.gcf())
