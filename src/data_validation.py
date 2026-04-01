import pandas as pd
import sys

def validate_data(filepath: str) -> pd.DataFrame:
    
    print("="*50)
    print("RUNNING DATA VALIDATION")
    print("="*50)

    # ── 1. Load ───────────────────────────────────────
    df = pd.read_csv(filepath)
    print(f"\n✅ File loaded | Shape: {df.shape}")

    # ── 2. Expected columns check ─────────────────────
    expected_cols = [
        'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education',
        'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome',
        'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
        'Property_Area', 'Loan_Status'
    ]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ FAIL - Missing columns: {missing_cols}")
        sys.exit(1)  # hard stop - data is unusable
    else:
        print(f"✅ All expected columns present")

    # ── 3. Target column check ────────────────────────
    if 'Loan_Status' not in df.columns:
        print("\n❌ FAIL - Target column 'Loan_Status' missing")
        sys.exit(1)
    else:
        print(f"✅ Target column exists")
        print(f"   Distribution:\n{df['Loan_Status'].value_counts()}")

    # ── 4. Missing value check ────────────────────────
    print("\n── Missing Values ──")
    missing_pct = (df.isnull().sum() / len(df)) * 100
    
    critical_cols = []  # >30% missing - hard fail
    warning_cols = []   # 5-30% missing - warn but continue

    for col, pct in missing_pct.items():
        if pct > 30:
            critical_cols.append((col, round(pct, 2)))
        elif pct > 5:
            warning_cols.append((col, round(pct, 2)))

    if critical_cols:
        print(f"❌ FAIL - These columns have >30% missing (unusable):")
        for col, pct in critical_cols:
            print(f"   {col}: {pct}%")
        sys.exit(1)
    
    if warning_cols:
        print(f"⚠️  WARNING - These columns have >5% missing (will impute):")
        for col, pct in warning_cols:
            print(f"   {col}: {pct}%")
    else:
        print("✅ No significant missing values")

    # ── 5. Minimum rows check ─────────────────────────
    if len(df) < 100:
        print(f"\n❌ FAIL - Only {len(df)} rows, too little to train")
        sys.exit(1)
    else:
        print(f"\n✅ Row count OK: {len(df)} rows")

    # ── 6. Duplicate check ────────────────────────────
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"⚠️  WARNING - {dupes} duplicate rows found")
    else:
        print(f"✅ No duplicates")

    print("\n✅ VALIDATION PASSED - Data is good to go\n")
    return df


if __name__ == "__main__":
    df = validate_data("data/raw/train_u6lujuX_CVtuZ9i.csv")