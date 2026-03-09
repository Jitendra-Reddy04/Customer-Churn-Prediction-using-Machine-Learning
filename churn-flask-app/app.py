from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/Rf_churn_model.sav", "rb"))

# Load reference dataset
df_ref = pd.read_csv("tele_churn_1.csv")

# Store last prediction
last_prediction = None
last_probability = None

MODEL_FEATURES = [
    'SeniorCitizen','MonthlyCharges','TotalCharges','gender','Partner','Dependents',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod','tenure'
]


@app.route("/", methods=["GET","POST"])
def home():

    global last_prediction, last_probability

    if request.method == "POST":

        # Collect form data
        data = []
        for i in range(1,20):
            val = request.form.get(f"query{i}")
            data.append(val)

        new_df = pd.DataFrame([data], columns=MODEL_FEATURES)

        # Convert numeric columns
        numeric_cols = ['SeniorCitizen','MonthlyCharges','TotalCharges','tenure']

        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        # Combine with reference dataset
        df_combined = pd.concat([df_ref, new_df], ignore_index=True)

        # One-hot encoding
        df_encoded = pd.get_dummies(df_combined, dtype=int)

        model_cols = model.feature_names_in_

        # Add missing columns
        for col in model_cols:
            if col not in df_encoded:
                df_encoded[col] = 0

        df_encoded = df_encoded[model_cols]

        # Predict
        prediction = model.predict(df_encoded.tail(1))[0]
        probability = model.predict_proba(df_encoded.tail(1))[:,1][0]

        prob = round(probability * 100, 2)

        # Save last prediction for dashboard
        last_prediction = int(prediction)
        last_probability = prob

        # Output message
        if prediction == 1:
            output1 = "⚠ Customer is likely to CHURN"
        else:
            output1 = "✅ Customer will STAY"

        output2 = f"Probability: {prob}%"

        return render_template(
            "result.html",
            output1=output1,
            output2=output2,
            probability=prob,
            churn=prediction
        )
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():

    df = pd.read_csv("tele_churn_1.csv")

    churn_counts = df['Churn'].value_counts().to_dict()
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