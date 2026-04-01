import pandas as pd
import numpy as np
import pickle


def load_artifacts():
    with open("artifacts/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("artifacts/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("artifacts/label_encoders.pkl", "rb") as f:
        label_encoders = pickle.load(f)
    with open("artifacts/ohe_encoder.pkl", "rb") as f:
        ohe = pickle.load(f)

    print("✅ All artifacts loaded")
    return model, scaler, label_encoders, ohe


def predict(input_dict, model, scaler, label_encoders, ohe):
    # ── 1. Convert input to dataframe ─────────────────
    df = pd.DataFrame([input_dict])

    # ── 2. Fix Dependents ─────────────────────────────
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].astype(float)

    # ── 3. Log Transform ──────────────────────────────
    log_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
    for col in log_cols:
        df[col] = np.log1p(df[col])

    # ── 4. Label Encode ───────────────────────────────
    le_cols = ['Gender', 'Married', 'Self_Employed', 'Education']
    for col in le_cols:
        df[col] = label_encoders[col].transform(df[col])

    # ── 5. One Hot Encode Property_Area ───────────────
    ohe_encoded = ohe.transform(df[['Property_Area']])
    ohe_cols = ohe.get_feature_names_out(['Property_Area'])
    ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_cols)
    df = pd.concat([df.drop('Property_Area', axis=1), ohe_df], axis=1)

    # ── 6. Scale Numerical Columns ────────────────────
    scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    df[scale_cols] = scaler.transform(df[scale_cols])

    # ── 7. Predict ────────────────────────────────────
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]

    result = "Approved" if prediction == 1 else "Rejected"
    confidence = round(probability[prediction] * 100, 2)

    print(f"\n── Prediction Result ──")
    print(f"  Decision   : {result}")
    print(f"  Confidence : {confidence}%")

    return result, confidence


if __name__ == "__main__":
    # ── Test with a sample applicant ──────────────────
    sample_applicant = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '2',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }

    model, scaler, label_encoders, ohe = load_artifacts()
    result, confidence = predict(sample_applicant, model, scaler, label_encoders, ohe)