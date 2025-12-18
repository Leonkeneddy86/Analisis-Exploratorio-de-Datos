# Analisis-Exploratorio-de-Datos
# Clinical Diabetes Risk & GLP-1 Dispensing Analysis
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Analytics](https://img.shields.io/badge/Analytics-EDA%20%7C%20Inference-informational)
![Stats](https://img.shields.io/badge/Stats-OLS%20%7C%20Logit-success)
![Domain](https://img.shields.io/badge/Domain-Healthcare-critical)
![Portfolio](https://img.shields.io/badge/Portfolio-Ready-brightgreen)  
Clinical Diabetes & Digital Health Analytics — Exploratory Data Analysis, statistical foundations and real-world prescription trends in digital health.

---

## 1. General Description

This project combines two complementary analytical perspectives to explore diabetes risk and digital health.

On one side, it uses a **synthetic clinical dataset for diabetes risk prediction**, built from medically validated ranges and distributions. Although the data do not represent real patients, they are suitable for exploratory analysis and machine-learning experimentation without compromising privacy.

On the other side, it incorporates **real-world NHS reimbursed prescription data** for GLP-1 weight-loss and diabetes-related medications such as Ozempic, Saxenda and Victoza. These records are aggregated at organisational level and reflect real pharmacological activity without individual patient information.

The two datasets are **not merged**. They are analysed independently to avoid incorrect inferences, while together providing a broader analytical view that links simulated clinical risk profiles with real medication-usage trends, always respecting the scope and limitations of each data source.

Beyond descriptive trends, the analysis explores risk discordance, metabolic profiles and behavioural factors to contextualise pharmacological uptake within broader public-health dynamics.
---
## Analytical overview

The diagram below summarises how the two independent datasets are analysed
and conceptually integrated within this project.

![Analytical overview diagram](assets/analysis_layers_diabetes_glp1.png)

*Two independent datasets are analysed separately and interpreted together.
No individual-level linkage is performed.*

## About the datasets used

This project is based on two complementary data sources:

1. **Synthetic clinical dataset — patient-level diabetes risk**  
   Source: Kaggle — *Health & Lifestyle Data for Diabetes Prediction*  
   https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction/data  

   Although the records do not represent real patients, the dataset follows medically validated ranges and distributions. This makes it suitable for exploratory analysis and machine-learning experimentation without compromising privacy.

2. **Real-world NHS dispensing dataset — GLP-1 medications (2019–2023)**  
   Source: Kaggle — *Weight Loss Medications*  
   https://www.kaggle.com/datasets/mpwolke/medications  

   The records reflect actual reimbursed prescriptions within the NHS, aggregated at organisational level and free of identifiable patient information.

**Important:** these datasets are analysed independently and are not merged. One is synthetic and population-wide, while the other captures real prescription activity over time. Together, they provide two complementary perspectives: simulated clinical profiles and real medication-usage trends.

---

## Column guide (plain English)

Units are included where standard; thresholds are used for interpretability, not for clinical diagnosis.
This section explains the **key columns used in the notebook** in non-technical terms.

### Dataset 1 — `diabetes_health_indicators_patientlevel_synthetic.csv` (synthetic patients)

**Demographics**
- `age` — Age in years.
- `gender` — Biological sex category (Male/Female/Other).
- `ethnicity` — Broad ethnic background category.
- `education_level` — Highest education attained.
- `income_level` — Income group category.
- `employment_status` — Employment category (e.g., Employed, Retired).

**Lifestyle**
- `smoking_status` — Smoking behaviour (Never/Former/Current).
- `alcohol_consumption_per_week` — Average number of alcoholic drinks per week.
- `physical_activity_minutes_per_week` — Weekly minutes of physical activity.
- `diet_score` — Diet quality score (higher values indicate healthier patterns).
- `sleep_hours_per_day` — Average sleep duration (hours/day).
- `screen_time_hours_per_day` — Daily screen time (hours/day).

**Medical history (binary flags)**
- `family_history_diabetes` — Family history of diabetes (0 = No, 1 = Yes).
- `hypertension_history` — History of hypertension (0 = No, 1 = Yes).
- `cardiovascular_history` — History of cardiovascular conditions (0 = No, 1 = Yes).

**Clinical measurements**
- `bmi` — Body Mass Index in kg/m² (body weight relative to height).
- `waist_to_hip_ratio` — Waist-to-hip ratio (central adiposity indicator).
- `systolic_bp` — Systolic blood pressure (mmHg).
- `diastolic_bp` — Diastolic blood pressure (mmHg).
- `heart_rate` — Resting heart rate (beats per minute).
- `cholesterol_total` — Total cholesterol (mg/dL).
- `hdl_cholesterol` — HDL (“good”) cholesterol (mg/dL).
- `ldl_cholesterol` — LDL (“bad”) cholesterol (mg/dL).
- `triglycerides` — Triglycerides (mg/dL).
- `glucose_fasting` — Fasting blood glucose (mg/dL).
- `glucose_postprandial` — Post-meal blood glucose (mg/dL).
- `insulin_level` — Blood insulin concentration (µU/mL).
- `hba1c` — HbA1c (%) as an indicator of longer-term blood glucose levels.

**Targets / outcomes used in modelling**
- `diabetes_risk_score` — Composite diabetes risk score provided in the dataset (used for regression).
- `diagnosed_diabetes` — Diabetes diagnosis flag (0 = No, 1 = Yes) (used for logistic regression).
- `diabetes_stage` — Stage label (e.g., No Diabetes, Pre-Diabetes, Type 1, Type 2, Gestational).

> Note: this dataset is synthetic and intended for analysis practice and modelling demonstrations. Values are designed to be clinically plausible but do not represent real individuals.

---

### Dataset 2 — `glp1_weight_loss_medications_dispensing_2019_2023.csv` (NHS dispensing)

Each row represents dispensing activity for a given month and presentation.

- `Year Month` — Month identifier in `YYYYMM` format (parsed to a proper date in the notebook).
- `Prescribed BNF Presentation Code` — Code for the prescribed presentation.
- `Prescribed BNF Presentation` — Text description of what was prescribed.
- `BNF Chemical Substance` — Active ingredient (molecule) name (e.g., Semaglutide, Liraglutide).
- `Dispensed (Reimbursed) BNF Presentation Code` — Code for the reimbursed dispensed presentation.
- `Dispensed (Reimbursed) BNF Presentation` — Text description of what was actually dispensed/reimbursed.
- `Items` — Count of dispensed items (a volume measure).
- `Total Quantity` — Total quantity dispensed (depends on presentation; used alongside `Items`).

**Added for accessibility**
- `Commercial Name` — Stakeholder-friendly brand name derived from `BNF Chemical Substance` (e.g., Ozempic, Victoza).  
  This supports readability for non-clinical audiences, while molecule-level analysis remains the clinically stable reference.

> Note: dispensing data are aggregated at organisational level and contain no patient-level linkage.

## 2. Project Objectives
- Understand clinical, lifestyle and demographic factors associated with diabetes risk.
- Apply rigorous exploratory data analysis techniques to healthcare data.
- Analyse real-world GLP-1 medication dispensing trends over time.
- Demonstrate statistical reasoning beyond descriptive metrics.
- Build a portfolio-ready healthcare analytics project aligned with digital health roles.
- Explicitly highlight limitations, assumptions and ethical considerations.
- Explore discordance between standard clinical indicators (e.g. BMI vs central adiposity) to highlight hidden risk profiles.
- Contextualise pharmacological trends alongside behavioural and metabolic factors, including ethical considerations around non-pharmacological alternatives.

---

## 3. Repository Structure

📦 rocio-perez-lopez/
┣ 📁 data/ # Original datasets (CSV)
┣ 📁 notebooks/ # Executed notebooks
┃ ┗ 📄 05_diabetes_risk_and_glp1_dispensing_eda_ipynb.ipynb
┣ 📄 requirements.txt # Python dependencies
┗ 📄 README.md # Project documentation

---

## 4. Technologies Used

- **Python** (NumPy, pandas, matplotlib, seaborn)
- **Statsmodels** (OLS, logistic regression, diagnostics)
- **scikit-learn** (predictive modelling)
- **Jupyter Notebook / Google Colab**
- **Excel** (initial inspection and enrichment)
- **GitHub**

---

## 5. How to Run the Project

1. Clone this repository.
2. Install dependencies from `requirements.txt`.
3. Place the datasets in the `data/` folder.
4. Open the notebook in Jupyter or Google Colab.
5. Run the notebook from top to bottom.

---

## 6. Data Cleaning and Transformation

Key preprocessing steps include:

- Parsing monthly time variables (`Year Month → datetime`).
- Renaming columns for clarity and clinical transparency.
- Creating derived features (BMI categories, activity levels, age bands).
- Converting variables to appropriate data types.
- Checking duplicates and missing values.
- Flagging outliers using IQR rules without automatic exclusion.
- Adding a **Commercial Name** column to the medication dataset to improve accessibility for non-clinical audiences, while retaining molecule-level analysis for clinical stability.

---

## 7. Analysis and Visualisations

The notebook contains an extensive set of well-justified visualisations, including:

- Univariate distributions of demographic and clinical variables.
- Time-series analysis of GLP-1 dispensing volumes.
- Molecule-level and brand-level comparisons.
- Bivariate clinical relationships, risk discordance patterns and regression-based inference.
- Correlation heatmaps of metabolic markers.
- Regression diagnostics and heteroscedasticity checks.
- Robust inference using Breusch–Pagan tests and HC3 standard errors.

Each visualisation addresses a specific clinical or public-health question and is explained in plain language.

---

## 8. Interpretation and Methodology

All findings are interpreted as **associations, not causal relationships**.

- The diabetes dataset is synthetic and cross-sectional.
- The dispensing dataset is real but aggregated at organisational level.
- There is no individual-level linkage between datasets.

Statistical models are used to quantify relationships, test hypotheses and assess uncertainty, while explicitly avoiding causal claims.
Rather than linking datasets mechanically, the project integrates them conceptually: patient-level metabolic risk patterns are interpreted alongside system-level prescribing responses, reflecting how clinical need, behavioural risk and policy-driven access interact in real healthcare settings.
---

## 9. Key Findings

- Diabetes risk increases consistently with age and BMI.
- Glycaemic markers show strong, clinically expected associations.
- Higher physical activity is associated with better glycaemic profiles.
- GLP-1 dispensing volumes increased markedly between 2019 and 2023.
- Robust statistical methods strengthen confidence in observed trends.
- Central adiposity indicators (waist-to-hip ratio) reveal risk patterns not fully captured by BMI alone.
- Lifestyle variables retain explanatory power alongside clinical markers, reinforcing the need to interpret pharmacological uptake within behavioural and ethical context.
---

## 10. Learnings

- A structured EDA workflow improves analytical reliability.
- Statistical significance must be interpreted alongside effect size and uncertainty.
- Diagnostic checks are essential in health data analysis.
- Clear communication and accessibility matter in clinical contexts.
- Ethical boundaries are part of good analytics practice.

---

## 11. References

- Health & Lifestyle Data for Diabetes Prediction (Synthetic)  
  https://www.kaggle.com/datasets/alamshihab075/health-and-lifestyle-data-for-diabetes-prediction/data

- Weight Loss Medications — NHS Prescriptions  
  https://www.kaggle.com/datasets/mpwolke/medications

- WHO guidelines on BMI and physical activity  
- Statsmodels and scikit-learn documentation

---
*Behind every dataset there is a health decision, and behind every decision, a responsibility.*
