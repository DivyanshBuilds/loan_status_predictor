import sys
import os

# This makes sure src/ folder is findable when running from root
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_validation import validate_data
from data_transformer import transform_data
from model_trainer import train_model

if __name__ == "__main__":

    print("="*50)
    print("STARTING LOAN STATUS PREDICTION PIPELINE")
    print("="*50)

    # ── Step 1: Validate ──────────────────────────────
    df = validate_data("data/raw/train_u6lujuX_CVtuZ9i.csv")

    # ── Step 2: Transform ─────────────────────────────
    X_train, X_test, y_train, y_test = transform_data(df)

    # ── Step 3: Train ─────────────────────────────────
    model = train_model(X_train, X_test, y_train, y_test)

    print("="*50)
    print("✅ PIPELINE COMPLETE")
    print("="*50)