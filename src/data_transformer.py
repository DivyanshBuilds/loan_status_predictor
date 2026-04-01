import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer


def transform_data(df: pd.DataFrame):

    print("="*50)
    print("RUNNING DATA TRANSFORMATION")
    print("="*50)

    # ── 1. Drop Loan_ID ───────────────────────────────
    df = df.drop('Loan_ID', axis=1)
    print("\n✅ Dropped Loan_ID")

    # ── 2. Fix Dependents (3+ → 3) ───────────────────
    df['Dependents'] = df['Dependents'].replace('3+', 3)
    df['Dependents'] = df['Dependents'].astype(float)  # float to handle NaN
    print("✅ Fixed Dependents column")

    # ── 3. Train/Test Split FIRST ─────────────────────
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"✅ Train/Test Split done | Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 4. Impute Missing Values ──────────────────────
    # Categorical columns → mode
    cat_impute_cols = ['Gender', 'Married', 'Self_Employed', 'Dependents']
    cat_imputers = {}

    for col in cat_impute_cols:
        imputer = SimpleImputer(strategy='most_frequent')
        X_train[[col]] = imputer.fit_transform(X_train[[col]])  # fit on train only
        X_test[[col]] = imputer.transform(X_test[[col]])        # apply to test
        cat_imputers[col] = imputer

    # Numerical columns → median
    num_impute_cols = ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    num_imputers = {}

    for col in num_impute_cols:
        imputer = SimpleImputer(strategy='median')
        X_train[[col]] = imputer.fit_transform(X_train[[col]])
        X_test[[col]] = imputer.transform(X_test[[col]])
        num_imputers[col] = imputer

    print("✅ Imputation done (fit on train, applied to both)")

    # ── 5. Log Transform Skewed Columns ──────────────
    log_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

    for col in log_cols:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    print("✅ Log transform applied to skewed columns")

    # ── 6. Label Encode Binary Categorical Columns ────
    le_cols = ['Gender', 'Married', 'Self_Employed', 'Education']
    label_encoders = {}

    for col in le_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])   # fit on train only
        X_test[col] = le.transform(X_test[col])          # apply to test
        label_encoders[col] = le                          # save encoder

    # Encode target
    le_target = LabelEncoder()
    y_train = le_target.fit_transform(y_train)
    y_test = le_target.transform(y_test)
    label_encoders['Loan_Status'] = le_target

    print("✅ Label Encoding done")

    # ── 7. One Hot Encode Property_Area ──────────────
    ohe = OneHotEncoder(drop='first', sparse_output=False)
    
    ohe_train = ohe.fit_transform(X_train[['Property_Area']])   # fit on train only
    ohe_test = ohe.transform(X_test[['Property_Area']])          # apply to test

    ohe_cols = ohe.get_feature_names_out(['Property_Area'])
    ohe_train_df = pd.DataFrame(ohe_train, columns=ohe_cols, index=X_train.index)
    ohe_test_df = pd.DataFrame(ohe_test, columns=ohe_cols, index=X_test.index)

    X_train = pd.concat([X_train.drop('Property_Area', axis=1), ohe_train_df], axis=1)
    X_test = pd.concat([X_test.drop('Property_Area', axis=1), ohe_test_df], axis=1)

    print("✅ One Hot Encoding done for Property_Area")

    # ── 8. Scale Numerical Columns ────────────────────
    scale_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    scaler = StandardScaler()

    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])   # fit on train only
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])          # apply to test

    print("✅ Scaling done (fit on train, applied to both)")

    # ── 9. Save Artifacts ─────────────────────────────
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Save train and test CSVs
    train_df = X_train.copy()
    train_df['Loan_Status'] = y_train
    test_df = X_test.copy()
    test_df['Loan_Status'] = y_test

    train_df.to_csv("data/processed/train_final.csv", index=False)
    test_df.to_csv("data/processed/test_final.csv", index=False)

    # Save encoders and scaler for future use (production)
    with open("artifacts/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
    with open("artifacts/ohe_encoder.pkl", "wb") as f:
        pickle.dump(ohe, f)
    with open("artifacts/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("✅ Train/Test CSVs saved to data/processed/")
    print("✅ Encoders and scaler saved to artifacts/")
    print("\n✅ TRANSFORMATION COMPLETE\n")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from data_validation import validate_data
    df = validate_data("data/raw/train_u6lujuX_CVtuZ9i.csv")
    X_train, X_test, y_train, y_test = transform_data(df)
    print(f"Final shapes → X_train: {X_train.shape} | X_test: {X_test.shape}")