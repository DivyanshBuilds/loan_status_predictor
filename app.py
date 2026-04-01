import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, render_template, request
from predictor import load_artifacts, predict

app = Flask(__name__)

# Load artifacts once when server starts
model, scaler, label_encoders, ohe = load_artifacts()

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None
    error = None

    if request.method == 'POST':
        try:
            input_dict = {
                'Gender':             request.form['Gender'],
                'Married':            request.form['Married'],
                'Dependents':         request.form['Dependents'],
                'Education':          request.form['Education'],
                'Self_Employed':      request.form['Self_Employed'],
                'ApplicantIncome':    float(request.form['ApplicantIncome']),
                'CoapplicantIncome':  float(request.form['CoapplicantIncome']),
                'LoanAmount':         float(request.form['LoanAmount']),
                'Loan_Amount_Term':   float(request.form['Loan_Amount_Term']),
                'Credit_History':     float(request.form['Credit_History']),
                'Property_Area':      request.form['Property_Area']
            }

            result, confidence = predict(input_dict, model, scaler, label_encoders, ohe)

        except Exception as e:
            error = f"Something went wrong: {str(e)}"

    return render_template('index.html', result=result, confidence=confidence, error=error)


if __name__ == '__main__':
    app.run(debug=True)