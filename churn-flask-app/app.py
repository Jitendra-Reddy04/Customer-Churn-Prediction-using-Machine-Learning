from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/Rf_churn_model.sav", "rb"))

# Load reference dataset
df_ref = pd.read_csv("tele_churn_1.csv")

# Store last prediction for dashboard
last_prediction = None
last_probability = None

# NEW: prediction counters for dashboard
stay_count = 0
churn_count = 0

MODEL_FEATURES = [
    'SeniorCitizen','MonthlyCharges','TotalCharges','gender','Partner','Dependents',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod','tenure'
]

@app.route("/", methods=["GET","POST"])
def home():

    global last_prediction, last_probability
    global stay_count, churn_count

    if request.method == "POST":

        data = []

        for i in range(1,20):
            val = request.form.get(f"query{i}")
            data.append(val)

        new_df = pd.DataFrame([data], columns=MODEL_FEATURES)

        numeric_cols = ['SeniorCitizen','MonthlyCharges','TotalCharges','tenure']

        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        # Combine with reference dataset
        df_combined = pd.concat([df_ref, new_df], ignore_index=True)

        # One hot encoding
        df_encoded = pd.get_dummies(df_combined, dtype=int)

        model_cols = model.feature_names_in_

        for col in model_cols:
            if col not in df_encoded:
                df_encoded[col] = 0

        df_encoded = df_encoded[model_cols]

        # Prediction
        prediction = model.predict(df_encoded.tail(1))[0]

        probs = model.predict_proba(df_encoded.tail(1))[0]

        stay_prob = probs[0] * 100
        churn_prob = probs[1] * 100

        if prediction == 1:
            output1 = "⚠ Customer is likely to CHURN"
            confidence = round(churn_prob,2)
            churn_count += 1
        else:
            output1 = "✅ Customer will STAY"
            confidence = round(stay_prob,2)
            stay_count += 1

        output2 = f"Prediction Confidence: {confidence}%"

        last_prediction = int(prediction)
        last_probability = confidence

        return render_template(
            "result.html",
            output1=output1,
            output2=output2,
            probability=round(churn_prob,2),
            stay=round(stay_prob,2),
            churn=prediction
        )

    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    # Dynamic prediction distribution
    churn_counts = {
        "Stayed": stay_count,
        "Churned": churn_count
    }

    df = pd.read_csv("tele_churn_1.csv")

    contract_counts = df['Contract'].value_counts().to_dict()
    internet_counts = df['InternetService'].value_counts().to_dict()
    monthly = df['MonthlyCharges'].tolist()

    return render_template(
        "dashboard.html",
        churn=churn_counts,
        contract=contract_counts,
        internet=internet_counts,
        monthly=monthly,
        last_prediction=last_prediction,
        last_probability=last_probability
    )

if __name__ == "__main__":
    app.run(debug=True)