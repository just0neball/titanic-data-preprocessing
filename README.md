# Titanic Data Preprocessing

**Codveda Technologies | Machine Learning Internship | Level 1 — Task 1**

## Overview

This project implements a reproducible data preprocessing pipeline for the Titanic passenger dataset using Python and scikit-learn.

The objective is to transform raw passenger records into clean, numerical, and model-ready features while preventing data leakage.

## Dataset

**Source:** [Titanic — Machine Learning from Disaster (Kaggle)](https://www.kaggle.com/competitions/titanic/data)

The dataset contains **891 passenger records and 12 columns**, including demographic information, ticket details, and survival outcomes.

### Initial Data Quality Assessment

| Feature | Missing Values | Missing % |
|---|---:|---:|
| Cabin | 687 | 77.10% |
| Age | 177 | 19.87% |
| Embarked | 2 | 0.22% |

Exploratory analysis also identified a strongly right-skewed Fare distribution, with a mean of 32.20 and a median of 14.45.

## Methodology

### 1. Feature Selection

Seven features were selected for the baseline preprocessing workflow.

- **Numerical:** Age, SibSp, Parch, Fare
- **Categorical:** Sex, Embarked, Pclass
- **Target:** Survived

PassengerId was excluded as an identifier. Name and Ticket were excluded because they require additional feature engineering, while Cabin was excluded due to its high missing-value rate and the scope of the task.

These exclusions do not imply that the features have no predictive value.

### 2. Train / Validation Split

The labeled dataset was divided using an **80/20 stratified split** with `random_state=42`.

| Dataset | Samples | Survival Rate |
|---|---:|---:|
| Training | 712 | 38.34% |
| Validation | 179 | 38.55% |

The held-out validation set fulfills the task's train/test split requirement. Kaggle's separate competition test dataset was not used because its survival labels are unavailable.

### 3. Preprocessing Pipeline

A reusable preprocessing workflow was implemented using `Pipeline` and `ColumnTransformer`.

**Numerical features**
- Median imputation for missing values
- Standardization using `StandardScaler`

**Categorical features**
- Most-frequent imputation for missing values
- One-Hot Encoding with unknown-category handling

All preprocessing parameters were fitted exclusively on the training data and subsequently applied to the validation set without refitting.

## Results

| Metric | Result |
|---|---|
| Original Features Selected | 7 |
| Features After Encoding | 12 |
| Processed Training Shape | 712 × 12 |
| Processed Validation Shape | 179 × 12 |
| Missing Values After Processing | 0 |
| Preprocessing Validation Checks | 8/8 Passed |
| Exported Artifacts | Integrity Checks Passed |

The fitted numerical imputer used a training-set Age median of **28.5**, while the categorical imputer used **S** for missing Embarked values.

Both processed datasets contain consistent numerical features, with preserved row alignment and verified target values.

## Project Structure

- **notebooks/**
  - `titanic_data_preprocessing.ipynb` — Data exploration, preprocessing, and validation
- **outputs/**
  - `train_processed.csv` — Processed training data
  - `validation_processed.csv` — Processed validation data
  - `preprocessor.joblib` — Fitted preprocessing pipeline
- **README.md** — Project documentation
- **requirements.txt** — Python dependencies
- **.gitignore** — Excluded local and temporary files

## How to Run

**Kaggle**

1. Open the notebook on Kaggle.
2. Add the Titanic competition dataset.
3. Run all cells in sequence.
4. Access the generated artifacts under `/kaggle/working/titanic_preprocessing/`.

**Local Environment**

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Place the original Kaggle `train.csv` in the repository root.
4. Launch Jupyter from the repository root and run the notebook.

Locally generated artifacts are saved under `outputs/titanic_preprocessing/`.

## Technologies

Python · pandas · scikit-learn · Matplotlib · Seaborn · joblib · Jupyter Notebook · Kaggle

## Limitations & Future Improvements

This project focuses on baseline preprocessing rather than predictive modeling.

Potential improvements include feature engineering from Name, Ticket, and Cabin, alternative treatments for skewed Fare values, and missing-value indicators for Age.

These approaches would require further evaluation before adoption.

## Conclusion

The project successfully transformed raw Titanic passenger data into consistent, numerical, model-ready features using a reusable preprocessing pipeline.

All implemented validation and exported-artifact integrity checks passed. The resulting datasets and fitted pipeline are ready for downstream machine learning experiments.

---
**Level 1 | Task 1: Data Preprocessing for Machine Learning**
