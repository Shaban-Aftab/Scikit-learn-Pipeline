# 📊 End-to-End Customer Churn Machine Learning Pipeline

A modular, production-ready machine learning system that uses **scikit-learn's Pipeline API** to predict customer churn on the **IBM Telco Churn Dataset**. It features robust numerical/categorical preprocessing branches, data-cleaning guards, multi-model hyperparameter search (`GridSearchCV` evaluating Logistic Regression vs. Random Forest), model serialization using `joblib`, and an interactive Streamlit serving dashboard.

🚀 **GitHub Repository:** [Shaban-Aftab/News-Topic-Classifier-BERT-FineTuned-](https://github.com/Shaban-Aftab/News-Topic-Classifier-BERT-FineTuned-) (Or your active pipeline repo)

---

## 🎨 Features & Architecture

* **scikit-learn Pipeline API**: Prevents data leakage during cross-validation by wrapping all imputing, scaling, encoding, and estimator fitting into a unified, modular execution sequence.
* **ColumnTransformer Branching**:
  * **Numerical Pipeline**: Median Imputation + `StandardScaler` (Applied to `tenure`, `MonthlyCharges`, `TotalCharges`).
  * **Categorical Pipeline**: Most Frequent Imputation + `OneHotEncoder(handle_unknown='ignore')` (Ensures robust handling of unseen labels during serving).
* **Data Cleaning Gotcha Guard**: Automatic cleaning of `TotalCharges` (coercing blank strings to `NaN` for new customers where `tenure = 0`) and mapping of `SeniorCitizen` categorical values.
* **Multi-Model GridSearchCV**: Searches across multiple model classes (Logistic Regression vs. Random Forest) and their hyperparameter grids in a single joint optimization pass, maximizing the test **F1-score**.
* **Streamlit Serving Dashboard**: Features an interactive UI with sliders, dropdown inputs, churn risk metrics, gauge status markers, and explainable AI structural risk factors.

---

## 📂 Project Structure

```
├── customer_churn_pipeline.ipynb  # Self-contained Jupyter Notebook for Kaggle execution
├── pipeline.py                    # Shared data preprocessor & cleaning pipelines
├── train.py                       # Training module with multi-model Grid Search & joblib saving
├── app.py                         # Streamlit interactive UI application
├── requirements.txt               # Dependencies file
└── README.md                      # Detailed documentation
```

---

## ⚡ Setup & Deployment

### 1. Local Setup
Clone the repository and install dependencies:

```bash
git clone <your-repo-url>
cd "Scikit-learn Pipeline"
pip install -r requirements.txt
```

### 2. Streamlit Servable Web App
Execute the app locally or run it directly on Streamlit Community Cloud:

```bash
streamlit run app.py
```

* **Note**: Make sure to drop the trained `customer_churn_model.joblib` file (generated from your Kaggle run) into the root folder to load the model.

---

## 🎯 Kaggle Integration

Because training models can take considerable compute time, you should run the **`customer_churn_pipeline.ipynb`** notebook directly on Kaggle:

1. **Create a Kaggle Notebook** and import `customer_churn_pipeline.ipynb`.
2. **Execute all cells** to download the dataset, clean the schema, execute the multi-model Grid Search, and print metric curves.
3. **Download** the generated `customer_churn_model.joblib` from the Kaggle Output section.
4. **Push** it to your GitHub repository to serve the model live on Streamlit Cloud!
