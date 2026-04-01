# Loan Status Predictor

A machine learning pipeline that predicts whether a loan application will be approved or rejected based on applicant details.

---

## Project Structure
```
loan_status_predictor/
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Transformed train/test CSVs
├── artifacts/                      # Saved model, scaler, encoders
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── ohe_encoder.pkl
├── src/
│   ├── data_validation.py          # Load and validate raw data
│   ├── data_transformer.py         # Clean, encode, scale, split
│   ├── model_trainer.py            # Train, evaluate, save model
│   └── predictor.py                # Load model and predict on new data
├── notebook/
│   └── eda.ipynb                   # Exploratory data analysis
├── main.py                         # Runs full training pipeline
├── requirements.txt
└── README.md
```

---

## How It Works

### Training Pipeline
Run once to train and save the model:
```
main.py → data_validation.py → data_transformer.py → model_trainer.py
```

### Inference Pipeline
Run to predict on a new applicant:
```
predictor.py → load artifacts → transform input → model → Approved/Rejected
```

---

## How To Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/loan_status_predictor.git
cd loan_status_predictor
```

### 2. Create and activate conda environment
```bash
conda create -n loan_predictor python=3.10.20
conda activate loan_predictor
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the training pipeline
```bash
python main.py
```

### 5. Test prediction on a new applicant
```bash
python src/predictor.py
```

---

## Model

| Parameter | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 100 |
| max_depth | 10 |
| max_features | sqrt |
| class_weight | balanced |
| random_state | 42 |

---

## Results

### Cross Validation (5 Fold)
| Metric | Score |
|---|---|
| F1 | 0.851 ± 0.028 |
| Precision | 0.780 ± 0.034 |
| Recall | 0.938 ± 0.040 |
| ROC-AUC | 0.751 ± 0.068 |

### Test Set
| Metric | Score |
|---|---|
| F1 | 0.883 |
| Precision | 0.840 |
| Recall | 0.929 |
| ROC-AUC | 0.799 |

---

## Dataset

- Source: Loan Prediction Dataset
- Rows: 614
- Features: 12 (after dropping Loan_ID)
- Target: Loan_Status (Y/N)
- Class distribution: 69% Approved, 31% Rejected

---

## Built By

Divyansh — Building end-to-end ML pipelines and exploring data-driven problem solving in real-world applications.