# Fairness via Explanation Quality 📊

This repository investigates the disparities in the quality of post-hoc explanations generated for complex black-box models, specifically focusing on the influence of fairness constraints on these explanations across diverse demographic groups. The work is inspired by recent research highlighting the importance of ensuring that explanations are not only accurate but also fair, meaning that explanation quality should be consistently high across different population subgroups, such as gender, age, or race

**Repository Overview**

This repository contains code and resources for evaluating disparities in the quality of post hoc explanations provided by machine learning models, with a focus on fairness across demographic groups


## Purpose

The main goal of this project is to:
- **Evaluate the quality of post hoc explanations** (e.g., LIME, KernelSHAP) for black-box models under various fairness constraints.
- **Identify and measure disparities** in explanation quality (e.g., fidelity, consistency, stability) between demographic groups.
- **Investigate the impact of model unfairness** on the fairness and reliability of explanations.

## 🎯 Motivation
This work is motivated by the increasing use of machine learning in high-stakes domains (e.g., healthcare, criminal justice), where unfair or unreliable explanations can have serious real-world consequences.

## ✅ Key Features

- **Fairness-aware model training:** Uses the Exponentiated Gradient algorithm to enforce fairness constraints (e.g., demographic parity) during model training.
- **Explanation generation:** Supports popular post hoc explanation methods such as LIME and KernelSHAP.
- **Quality metrics:** Implements metrics for evaluating explanation quality, including:
  - **Fidelity:** How accurately the explanation matches the black-box model's predictions.
  - **Consistency:** How similar explanations are for repeated runs on the same input.
  - **Stability:** How robust explanations are to small input perturbations.
- **Disparity measurement:** Quantifies differences in explanation quality across demographic groups using statistical tests (e.g., Mann-Whitney U test).

## Datasets

The code is designed to work with standard benchmark datasets relevant to fairness research, such as:
- **ACSIncome** (income prediction, gender as sensitive attribute)
- **ACSEmployment** (employment prediction, age as sensitive attribute)
- **COMPAS** (recidivism prediction, race as sensitive attribute)

## How It Works

1. **Data Preparation:** Datasets are split into training, testing, and explanation sets to ensure balanced evaluation.
2. **Model Training:** Black-box models (e.g., XGBoost, Random Forest) are trained, with optional fairness constraints.
3. **Explanation Generation:** Post hoc explanations are generated for model predictions using LIME or KernelSHAP.
4. **Quality Evaluation:** Explanation quality is measured for each demographic subgroup using fidelity, consistency, and stability metrics.
5. **Disparity Analysis:** Statistical tests are performed to identify significant differences in explanation quality between groups.

## Results

- **Explanation quality disparities exist:** Especially for complex, non-linear models and in datasets with imbalanced or underrepresented subgroups.
- **Fairness constraints help but are not a panacea:** While enforcing fairness can reduce disparities, it does not always guarantee equitable explanation quality across all groups.
- **Context matters:** The choice of fairness metric and the characteristics of the dataset influence the observed disparities.

## Running Experiments
The Slurm batch script (`run_experiments.sh`) executes a series of experiments.
- **For each dataset, model, and sensitive attribute group**, the script launches a parallel job.
  - **`"ACSIncome"`**: Uses `gender=("male" "female")` for gender groups.
  - **`"ACSEmployment"`**: Uses `age=("g0" "g1")` for age groups.
  - **`"COMPAS"`**: Uses `race=("afr" "cau")` for race groups (where `"afr"` is African-American, `"cau"` is Caucasian).
- **`srun python ...`**: Runs the main experiment script in parallel for each combination, with all specified seeds.


## Usage

To use this repository, follow these steps:
```bash
# 1. Clone the repository
git clone https://github.com/DaveMbiazi/Fairness-via-Explanation-Quality.git
cd Fairness-via-Explanation-Quality

# 2. Create and activate the environment
python3 -m venv fairenv
source fairenv/bin/activate

# 3. Install any remaining dependencies
pip install -r requirements.txt