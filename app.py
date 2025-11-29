import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ==============================
# 🔹 MODEL LOAD
# ==============================
MODEL_PATH = 'artifacts/model_data.joblib'

@st.cache_resource
def load_model():
    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"], data["features"], data["cols_to_scale"]

model, scaler, features, cols_to_scale = load_model()


# ==============================
# 🔹 DATA PREPARATION
# ==============================
def prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    input_data = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency,

        # Encoding categorical variables
        'residence_type_Owned': 1 if residence_type == 'Owned' else 0,
        'residence_type_Rented': 1 if residence_type == 'Rented' else 0,

        'loan_purpose_Education': 1 if loan_purpose == 'Education' else 0,
        'loan_purpose_Home': 1 if loan_purpose == 'Home' else 0,
        'loan_purpose_Personal': 1 if loan_purpose == 'Personal' else 0,

        'loan_type_Unsecured': 1 if loan_type == 'Unsecured' else 0,

        # Dummy features required for your model
        'number_of_dependants': 1,
        'years_at_current_address': 1,
        'zipcode': 1,
        'sanction_amount': 1,
        'processing_fee': 1,
        'gst': 1,
        'net_disbursement': 1,
        'principal_outstanding': 1,
        'bank_balance_at_application': 1,
        'number_of_closed_accounts': 1,
        'enquiry_count': 1
    }

    df = pd.DataFrame([input_data])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]

    return df


# ==============================
# 🔹 CREDIT SCORE CALCULATION
# ==============================
def calculate_credit_score(input_df, base_score=300, scale_lenth=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_

    default_probability = 1 / (1+np.exp(-x))
    non_default_probability = 1 - default_probability

    credit_score = base_score + non_default_probability.flatten() * scale_lenth

    def get_rating(score):
        if 300 <= score < 500: return 'Poor'
        elif 500 <= score < 650: return 'Average'
        elif 650 <= score < 750: return 'Good'
        elif 750 <= score <= 900: return 'Excellent'
        else: return 'Undefined'

    rating = get_rating(credit_score[0])

    return default_probability.flatten()[0], int(credit_score), rating



# ==============================
# 🔹 PREDICTION FUNCTION
# ==============================
def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    input_df = prepare_df(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                          delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                          residence_type, loan_purpose, loan_type)

    probability, credit_score, rating = calculate_credit_score(input_df)
    return probability, credit_score, rating



# ==================================================
# 🔹 STREAMLIT UI (Your original main.py code below)
# ==================================================
st.set_page_config(page_title="Udhay Finance: Credit Risk Modelling", page_icon="📊")
st.title("Udhay Finance: Credit Risk Modelling")


row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)

with row1[0]:
    age = st.number_input('Age', min_value=18, step=1, max_value=100, value=28)
with row1[1]:
    income = st.number_input('Income', min_value=0, value=1200000)
with row1[2]:
    loan_amount = st.number_input('Loan Amount', min_value=0, value=2560000)

loan_to_income_ratio = loan_amount / income if income > 0 else 0
with row2[0]:
    st.text("Loan to Income Ratio:")
    st.text(f"{loan_to_income_ratio:.2f}")

with row2[1]:
    loan_tenure_months = st.number_input('Loan Tenure (months)', min_value=0, step=1, value=36)
with row2[2]:
    avg_dpd_per_delinquency = st.number_input('Avg DPD', min_value=0, value=20)

with row3[0]:
    delinquency_ratio = st.number_input('Delinquency Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[1]:
    credit_utilization_ratio = st.number_input('Credit Utilization Ratio', min_value=0, max_value=100, step=1, value=30)
with row3[2]:
    num_open_accounts = st.number_input('Open Loan Accounts', min_value=1, max_value=4, step=1, value=2)

with row4[0]:
    residence_type = st.selectbox('Residence Type', ['Owned', 'Rented', 'Mortgage'])
with row4[1]:
    loan_purpose = st.selectbox('Loan Purpose', ['Education', 'Home', 'Auto', 'Personal'])
with row4[2]:
    loan_type = st.selectbox('Loan Type', ['Unsecured', 'Secured'])


if st.button('Calculate Risk'):
    probability, credit_score, rating = predict(age, income, loan_amount, loan_tenure_months,
                                                avg_dpd_per_delinquency, delinquency_ratio,
                                                credit_utilization_ratio, num_open_accounts,
                                                residence_type, loan_purpose, loan_type)

    st.write(f"📉 Default Probability: **{probability:.2%}**")
    st.write(f"💳 Credit Score: **{credit_score}**")
    st.write(f"🏷 Rating: **{rating}**")
