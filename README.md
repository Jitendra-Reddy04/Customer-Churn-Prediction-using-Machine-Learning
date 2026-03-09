# 📞 Customer Churn Prediction

Predict customer churn for a telecom company using machine learning and a user-friendly Flask web app.

---

## 🚀 Project Overview

This project uses a real-world telecom dataset to predict whether a customer will churn (leave the service) or not. It includes:

- **Data Cleaning & Preprocessing**
- **Exploratory Data Analysis (EDA) with Visualizations**
- **Model Building & Evaluation**
- **A Flask Web Application for Live Predictions**

---

## 🗂️ Project Structure

```
Churn_practice/
│
├── Churn_model_building.ipynb      # Model training and evaluation notebook
├── Churn_preedict_Analysis.ipynb   # Data cleaning and EDA notebook
├── churn-flask-app/
│   ├── app.py                      # Flask backend
│   ├── model/
│   │   └── Rf_churn_model.sav      # Trained RandomForest model
│   ├── static/
│   │   └── style.css               # CSS for frontend
│   ├── templates/
│   │   └── home.html               # HTML frontend
│   └── tele_churn_1.csv            # Reference data for preprocessing
├── tel_churn.csv                   # Raw dataset
└── README.md                       # Project documentation
```

---

## 📊 Exploratory Data Analysis

- **Class Imbalance:**  
  ![Churn Distribution](https://github.com/chanducn/Churn_Prediction/blob/ae774669ac699847d1e9dde064fa894e42615c70/Churn_imbalance.png)
- **Monthly Charges vs Churn:**  
  ![Monthly Charges KDE](https://github.com/chanducn/Churn_Prediction/blob/ae774669ac699847d1e9dde064fa894e42615c70/monthly%20charges%20by%20churn.png)
- **Correlation Heatmap:**  
  ![Correlation Heatmap](https://github.com/chanducn/Churn_Prediction/blob/ae774669ac699847d1e9dde064fa894e42615c70/corelation%20heatmap.png)

*Visualizations generated in the EDA notebook.*

---

## 🛠️ Model Building

- **Preprocessing:**  
  - Handled missing values
  - Converted categorical variables to dummies
  - Binned tenure into groups
  - Addressed class imbalance with SMOTEENN

- **Models Tried:**  
  - Decision Tree
  - Random Forest (Best)
  - Logistic Regression
  - KNN, SVM, Naive Bayes

- **Best Model:**  
  Random Forest Classifier  
  *Exported as `Rf_churn_model.sav` for deployment.*

---

## 🌐 Flask Web Application

- **User Input:**  
  Enter customer details via a web form.
- **Prediction:**  
  The app preprocesses input, applies the trained model, and displays churn prediction with confidence.



## 🏁 How to Run Locally

1. **Clone the repository:**
    ```bash
    git clone https://github.com/chanducn/Churn_Prediction.git
    cd Churn_practice/churn-flask-app
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Flask app:**
    ```bash
    python app.py
    ```

4. **Open your browser:**  
   Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📝 Key Files

- `Churn_model_building.ipynb` – Model training and export
- `Churn_preedict_Analysis.ipynb` – Data cleaning and EDA
- `churn-flask-app/app.py` – Flask backend
- `churn-flask-app/templates/index.html` – Web form
- `churn-flask-app/static/style.css` – Styling

---

## 📚 References

- [Kaggle: Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/)

---

## 🙌 Acknowledgements

Thanks to the open-source community and all contributors!

---

## 📧 Contact

For questions or suggestions, open an issue or contact (mailto:sonu.aps1998@gmail.com).
