from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load model
model = pickle.load(open("model/Rf_churn_model.sav", "rb"))

# Load dataset
df_ref = pd.read_csv("tele_churn_1.csv")

# Global variables
last_prediction = None
last_probability = None
last_input = None
last_suggestions = []
last_insights = []

stay_count = 0
churn_count = 0

MODEL_FEATURES = [
    'SeniorCitizen','MonthlyCharges','TotalCharges','gender','Partner','Dependents',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract',
    'PaperlessBilling','PaymentMethod','tenure'
]

# =========================
# 🔥 INSIGHTS FUNCTION
# =========================
def generate_insights(data, prediction):

    insights = []

    try:
        tenure = float(data.get("tenure") or 0)
        charges = float(data.get("MonthlyCharges") or 0)
    except:
        tenure = 0
        charges = 0

    contract = data.get("Contract")
    security = data.get("OnlineSecurity")
    support = data.get("TechSupport")

    # 🔴 CHURN CASE
    if prediction == 1:

        if tenure < 12:
            insights.append("📉 Low tenure increases churn risk")

        if contract == "Month-to-month":
            insights.append("📄 Month-to-month contracts are unstable")

        if charges > 80:
            insights.append("💰 High monthly charges may reduce retention")

        if security == "No":
            insights.append("🔐 Lack of security services increases churn risk")

        if support == "No":
            insights.append("🛠 No tech support leads to dissatisfaction")

        if not insights:
            insights.append("⚠ Customer shows unexpected churn behavior")

    # 🟢 STAY CASE
    else:

        if tenure > 24:
            insights.append("📈 Long tenure improves customer loyalty")

        if contract in ["One year", "Two year"]:
            insights.append("📄 Long-term contract reduces churn")

        if security == "Yes":
            insights.append("🔐 Security services improve retention")

        if support == "Yes":
            insights.append("🛠 Tech support increases satisfaction")

        if charges < 80:
            insights.append("💰 Affordable pricing helps retention")

        if not insights:
            insights.append("✅ Customer profile is stable")

    return insights


# =========================
# 🏠 HOME ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():

    global last_prediction, last_probability, last_input
    global last_suggestions, last_insights
    global stay_count, churn_count

    if request.method == "POST":

        data = [request.form.get(f"query{i}") for i in range(1,20)]
        new_df = pd.DataFrame([data], columns=MODEL_FEATURES)

        # Convert numeric safely
        numeric_cols = ['SeniorCitizen','MonthlyCharges','TotalCharges','tenure']
        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        last_input = new_df.to_dict(orient="records")[0]

        # Encoding
        df_combined = pd.concat([df_ref, new_df], ignore_index=True)
        df_encoded = pd.get_dummies(df_combined)

        model_cols = model.feature_names_in_
        for col in model_cols:
            if col not in df_encoded:
                df_encoded[col] = 0

        df_encoded = df_encoded[model_cols]

        # Prediction
        prediction = model.predict(df_encoded.tail(1))[0]
        probs = model.predict_proba(df_encoded.tail(1))[0]

        stay_prob = round(probs[0]*100,2)
        churn_prob = round(probs[1]*100,2)

        if prediction == 1:
            output = "⚠ Customer is likely to CHURN"
            confidence = churn_prob
            churn_count += 1
        else:
            output = "✅ Customer will STAY"
            confidence = stay_prob
            stay_count += 1

        last_prediction = int(prediction)
        last_probability = confidence

        # =========================
        # 💡 SUGGESTIONS
        # =========================
        suggestions = []

        if new_df['Contract'][0] == "Month-to-month":
            suggestions.append("📄 Offer long-term contract plans")

        if new_df['MonthlyCharges'][0] > 80:
            suggestions.append("💰 Provide discounts or cheaper plans")

        if new_df['OnlineSecurity'][0] == "No":
            suggestions.append("🔐 Recommend security add-ons")

        if new_df['TechSupport'][0] == "No":
            suggestions.append("🛠 Offer technical support services")

        if new_df['tenure'][0] < 12:
            suggestions.append("🎁 Provide loyalty benefits")

        if new_df['PaymentMethod'][0] == "Electronic check":
            suggestions.append("💳 Encourage auto-payment methods")

        if new_df['InternetService'][0] == "Fiber optic" and churn_prob > 50:
            suggestions.append("📶 Improve service quality or pricing")

        if not suggestions:
            suggestions.append("✅ Customer is stable — maintain service quality")

        last_suggestions = suggestions

        # =========================
        # 🧠 INSIGHTS
        # =========================
        last_insights = generate_insights(last_input, last_prediction)

        return render_template(
            "result.html",
            output=output,
            confidence=confidence,
            churn=churn_prob,
            stay=stay_prob,
            suggestions=suggestions,
            insights=last_insights
        )

    return render_template("index.html")


# =========================
# 📊 DASHBOARD
# =========================
@app.route("/dashboard")
def dashboard():

    global last_prediction, last_probability
    global last_input, last_suggestions, last_insights

    churn_counts = {"Stayed": stay_count, "Churned": churn_count}

    # SAFE VALUES
    if last_prediction is not None:
        if last_prediction == 1:
            churn_val = last_probability
            stay_val = 100 - last_probability
        else:
            churn_val = 100 - last_probability
            stay_val = last_probability
    else:
        churn_val = 0
        stay_val = 0

    return render_template(
        "dashboard.html",
        churn=churn_counts,
        last_prediction=last_prediction,
        last_probability=last_probability,
        last_input=last_input,
        churn_val=churn_val,
        stay_val=stay_val,
        suggestions=last_suggestions,
        insights=last_insights
    )


# =========================
# ▶ RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)