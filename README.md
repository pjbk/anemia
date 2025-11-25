# 🩺 Interpretable Real-Time Anemia Risk Predictor

A clinical decision support system (CDSS) designed to diagnose anemia and assess its likelihood based on hematological and demographic parameters. Powered by a pre-trained **Stacking Ensemble Model (STEM)** and developed using **Streamlit**, this web-based application offers real-time predictions and interpretable explanations to assist healthcare professionals at the point of care. The model is trained on curated datasets from both clinical and open-source repositories, ensuring broad applicability and reliable performance.

---

## 🌐 Live Web App

 **Launch Now**: [https://anemia.streamlit.app/](https://anemia-dss.streamlit.app/)

![App Interface](https://github.com/pjbk/anemia-DSS/blob/main/anemia-predictor-interface.jpg)

---

## Key Features

- **Real-Time Diagnosis**: Predicts anemia status based on clinical input.
- **Model Interpretability**: Explains predictions with SHAP visualizations; highlights the top 5 contributing features.
- **Responsive UI**: Works seamlessly on both desktop and mobile devices.
- **Dark Mode Support**: Automatically adapts to the user's system theme.
- **Risk Probability**: Displays confidence score for each prediction.

---

## Dataset Sources

The model is trained using datasets from both local clinical sources and public repositories:

- **📁 Mendeley Dataset (Base Dataset)** (Aalok Healthcare Ltd., Dhaka, Bangladesh):  
  [https://data.mendeley.com/datasets/y7v7ff3wpj/1](https://data.mendeley.com/datasets/y7v7ff3wpj/1)

- **📁 Mendeley Dataset (Validation Dataset 1)** (Eureka diagnostic center, Lucknow, India):  
  [https://data.mendeley.com/datasets/dy9mfjchm7/1](https://data.mendeley.com/datasets/dy9mfjchm7/1)

- **📁 Kaggle Dataset (Validation Dataset 2)** (Open Source):  
  [https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset](https://www.kaggle.com/datasets/biswaranjanrao/anemia-dataset)

### References
1. Mojumdar, M.U., et al. *Pediatric Anemia Dataset: Hematological Indicators and Diagnostic Classification*. Mendeley Data, V1 (2024). [DOI](https://doi.org/10.17632/y7v7ff3wpj.1)  
2. Mojumdar, M.U., et al. *AnaDetect: An extensive dataset for advancing anemia detection, diagnostic methods, and predictive analytics in healthcare.* Data in Brief 58, 111195 (2025). [DOI](https://doi.org/10.1016/j.dib.2024.111195)
3. Vohra, Rajan; pahareeya, jankisharan; Hussain, Abir  (2021), “Complete Blood Count Anemia Diagnosis”, Mendeley Data, V1. [DOI] (https://doi.org/10.17632/dy9mfjchm7.1)
4. Vohra R, Hussain A, Dudyala AK, Pahareeya J, Khan W (2022) Multi-class classification algorithms for the diagnosis of anemia in an outpatient clinical setting. PLoS ONE 17(7): e0269685. [DOI] (https://doi.org/10.1371/journal.pone.0269685)

---

## Model Pipeline

```python
# model pipeline pseudocode  
# ANEMIA RISK PREDICTION PIPELINE
# Comprehensive machine learning pipeline for anemia risk stratification

Input: Dataset D with Observations O, Features X, Target y
Output: Processed Dataset, Model Evaluation, Confidence Intervals, 
        Permutation Scores, Nested CV Scores, Risk Stratification

## 1. DATA PREPROCESSING
def preprocess_data(D):
    # 1.1 Handle Missing Values
    for each feature Xi in X:
        missing_pct = (missing_count(Xi) / total_samples) * 100
        if missing_pct < 50%:
            D = impute_missing(D, Xi, method="mean", add_missing_flag=True)
        else:
            D = drop_feature(D, Xi)
    
    # 1.2 Encode Categorical Features
    for each categorical feature Xi:
        D = one_hot_encode(D, Xi)
    
    # 1.3 Remove Outliers (Z-score > 3)
    for each datapoint Xij in D:
        z_score = (Xij - mean(Xi)) / std(Xi)
        if abs(z_score) > 3:
            D = remove_outlier(D, Xij)
    
    return D

## 2. STATISTICAL FEATURE ANALYSIS
def analyze_features(D, X, y):
    results = {}
    for each feature Xi in X:
        # 2.1 Descriptive Statistics
        stats = {
            'mean': mean(Xi),
            'median': median(Xi), 
            'std': standard_deviation(Xi),
            'skewness': compute_skewness(Xi),
            'kurtosis': compute_kurtosis(Xi)
        }
        
        # 2.2 Statistical Significance Tests
        # T-test for normally distributed features
        t_stat, p_value_ttest = t_test(Xi[y==0], Xi[y==1])
        
        # Mann-Whitney U test for non-normal features
        u_stat, p_value_utest = mann_whitney_u(Xi[y==0], Xi[y==1])
        
        # 2.3 Independent Risk Score Calculation
        # Odds Ratio from univariate logistic regression
        risk_score = compute_odds_ratio(Xi, y)
        independent_risk_score = -log10(p_value_ttest) * risk_score
        
        # 2.4 SHAP Importance Score
        # Train temporary model for SHAP analysis
        temp_model = LogisticRegression()
        temp_model.fit(Xi.values.reshape(-1, 1), y)
        explainer = shap.Explainer(temp_model, Xi.values.reshape(-1, 1))
        shap_values = explainer(Xi.values.reshape(-1, 1))
        shap_importance_score = np.mean(np.abs(shap_values.values))
        
        # 2.5 Correlation Analysis
        correlations = {}
        for each Xj ≠ Xi:
            corr = pearson_correlation(Xi, Xj)
            correlations[Xj] = corr
        
        # 2.6 Feature Importance
        rf_importance = random_forest_importance(Xi)
        
        # Compile all results for feature Xi
        results[Xi] = {
            **stats, 
            'p_values': {
                't_test': p_value_ttest,
                'u_test': p_value_utest
            },
            'risk_scores': {
                'odds_ratio': risk_score,
                'independent_risk_score': independent_risk_score
            },
            'importance_scores': {
                'random_forest': rf_importance,
                'shap_importance': shap_importance_score
            },
            'correlations': correlations
        }
    
    return results

## 3. RISK STRATIFICATION ANALYSIS
def risk_stratification(D, X, y):
    # 3.1 Composite Risk Score: Σβᵢxᵢ
    composite_risk = sum(β_i * x_i for each feature i)
    
    # 3.2 Tertile Stratification
    risk_tertiles = {
        'Low': ≤ 33rd percentile,
        'Medium': 33rd–66th percentile,
        'High': > 66th percentile
    }
    
    # 3.3 Demographic Analysis
    for age_group in age_groups:
        for gender in genders:
            group_risk = compute_risk_score(age_group, gender)
            event_rate = compute_event_rate(age_group, gender)
    
    return risk_tertiles, composite_risk

## 4. MODEL DEFINITION & STACKING
def define_models():
    # Base Models
    models = {
        'DecisionTree': DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=3),
        'LogisticRegression': LogisticRegression(C=20, solver='liblinear', max_iter=1000),
        'SVM': SVC(C=10, kernel='linear', probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'NaiveBayes': GaussianNB(var_smoothing=2.88176e-10)
    }
    
    # Stacking Ensemble
    stack = StackingCVClassifier(
        classifiers=[dtc, lr, svc, knn, gnb],
        meta_classifier=SVC(C=10, kernel='linear', probability=True, random_state=42),
        cv=5, use_probas=True, random_state=42, shuffle=True,
        use_features_in_secondary=True
    )
    
    return models, stack

## 5. MODEL TRAINING & EVALUATION
def train_evaluate_models(D_processed):
    # 5.1 Stratified Train-Test Split
    D_train_val, D_test = stratified_split(D_processed, test_size=0.2, random_state=42)
    
    models, stack = define_models()
    results = {}
    
    # 5.2 Model Optimization & Evaluation
    for model_name, model in models.items():
        # Hyperparameter Tuning
        optimized_model = grid_search_tuning(model, D_train_val)
        
        # Cross-Validation
        cv_scores = cross_validate(optimized_model, D_train_val, cv=5, 
                                 scoring=['accuracy', 'precision', 'recall', 'f1'])
        
        # Final Test Evaluation
        test_score = evaluate(optimized_model, D_test)
        ci = bootstrap_confidence_interval(test_score, n_bootstraps=1000)
        perm_scores = permutation_test(optimized_model, D_test, n_permutations=1000)
        nested_cv = nested_cross_validation(optimized_model, D_processed)
        
        results[model_name] = {
            'cv_scores': cv_scores,
            'test_score': test_score,
            'confidence_interval': ci,
            'permutation_scores': perm_scores,
            'nested_cv': nested_cv
        }
    
    return results

## 6. MODEL EXPLAINABILITY
def explain_best_model(best_model, D_processed):
    # SHAP Analysis
    explainer = shap.Explainer(best_model)
    shap_values = explainer.shap_values(D_processed)
    
    # Feature Importance Plot
    shap.summary_plot(shap_values, D_processed)
    
    return shap_values

## 7. MAIN EXECUTION PIPELINE
def main_pipeline(D):
    # 7.1 Preprocess Data
    D_processed = preprocess_data(D)
    
    # 7.2 Statistical Analysis
    feature_analysis = analyze_features(D_processed, X, y)
    
    # 7.3 Risk Stratification
    risk_tertiles, composite_risk = risk_stratification(D_processed, X, y)
    
    # 7.4 Model Training & Evaluation
    model_results = train_evaluate_models(D_processed)
    
    # 7.5 Explain Best Model
    best_model = select_best_model(model_results)
    shap_explanation = explain_best_model(best_model, D_processed)
    
    # 7.6 Return Comprehensive Results
    return {
        'processed_data': D_processed,
        'feature_analysis': feature_analysis,
        'risk_stratification': risk_tertiles,
        'model_results': model_results,
        'best_model': best_model,
        'shap_explanation': shap_explanation
    }

# Execute Pipeline
final_results = main_pipeline(D)
```
---
## Quick Start
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pjbk/anemia.git
   cd anemia
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser at `http://localhost:8501`.
---

## Project Structure

```
dss-anemia/
├── README.md
├── app.py
├── STEM_model.pkl
├── scaler.pkl
└── requirements.txt
```
---
## User Guide
- **Input Data:** Provide patient's hematological and demographic data.
- **Prediction Output:** Receive anemia risk prediction and associated probability.
- **Visual Explanation:** View SHAP-based plot explaining the key contributing factors.
---
## Tools and Technologies
| Tool           | Purpose                                         |
| -------------- | ----------------------------------------------- |
| `Scikit-learn` | Machine learning models and evaluation          |
| `Streamlit`    | Interactive real-time web application interface |
| `SHAP`         | Interpretable model explanations (XAI)          |
| `Matplotlib`   | Data and SHAP visualization                     |

---
