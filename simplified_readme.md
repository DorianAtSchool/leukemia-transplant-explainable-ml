# Predictive Modeling for Leukemia Transplant Outcomes (Simplified)

## Overview
This project represents an **initial benchmark effort** in applying machine learning to stem cell transplant outcomes for leukemia patients. It is built on the **publicly available `ds1302` dataset** (~5,000 acute leukemia cases), making it reproducible and accessible for researchers.  

**Context**  
- **Leukemia** (AML/ALL) is an aggressive blood cancer that disrupts normal blood cell production.  
- A **stem cell transplant** (allogeneic hematopoietic cell transplantation, allo-HCT) can be curative, but success rates vary widely.  

**Motivation**  
- Predicting outcomes is difficult because they depend on many interacting factors: patient age, remission status, donor–recipient compatibility, and treatment regimens.  
- Machine learning offers tools to identify the most influential predictors and support clinical decision-making.  

**Current Stage (this project)**  
- Used the **`ds1302` public dataset** to train baseline ML models predicting overall survival (binary: alive vs dead).  
- Evaluated eight classifiers, analyzed feature importance, and identified key risk drivers.  
- Results provide a **first reproducible benchmark** and highlight unique challenges of clinical tabular data (structured missingness, censoring, collinearity).  

**Long-Term Goal**  
- Transition from binary survival prediction to **time-to-event survival modeling**.  
- Expand to **cause-specific outcomes** (e.g., relapse vs non-relapse mortality).  
- Ultimately **“flip the task”** to recommend the most suitable donor for a given patient — moving from risk assessment toward **precision donor selection**.  

---

## Abstract
Using the **public `ds1302` leukemia transplant dataset**, we evaluated eight machine learning classifiers to predict overall survival after allogeneic stem cell transplantation. The Support Vector Machine (SVM) performed best, achieving ~60% accuracy and an AUC of ~0.616. Although performance was modest, feature importance analysis consistently highlighted clinically meaningful factors:  

- **Age** — older patients had worse survival outcomes.  
- **Disease status at transplant** — being in remission strongly improved survival.  
- **GVHD prophylaxis regimens** — immune suppression strategies influenced outcomes.  
- **Donor–recipient compatibility (ABO match)** — showed modest but reproducible effects.  

These findings, while aligned with established clinical knowledge, demonstrate that ML can recover **interpretable and actionable insights** from real-world clinical datasets. This supports the potential of machine learning to eventually guide donor selection and treatment tailoring, provided richer features, survival modeling, and external validation are incorporated.  

---

## Data Processing
- **Original dataset**: 4,946 patients, 49 features  
- **After cleaning**: 4,653 patients, 20 features  
- **Target variable**: `dead` (1 = death, 0 = censored/alive)  
- **Cleaning decisions**:  
  - Dropped identifiers (patient IDs, centers)  
  - Removed outcome-based variables (to avoid leakage)  
  - Excluded variables with high missingness  
  - Rows with structured missing codes (99, –9) were dropped (to be improved later with imputation)  

---

## Model Performance
Eight classifiers benchmarked using 5-fold CV and GridSearchCV.  

**Best model: SVM**  
- Accuracy: **0.6015**  
- AUC: **~0.616**  
- Precision/Recall/F1: ~0.57–0.60  

**General outcome:** All models clustered around **AUC 0.56–0.61**, confirming this is a **difficult prediction task** with the given features.  

**Visual Placeholder:**  
![Model Performance](visuals/classification/svm/svm_confusion_matrix.png)  

---

## Feature Importance (Key Insights)
- **Age**: strongest, consistent predictor across all models.  
- **Disease status at transplant**: remission at transplant strongly improves survival.  
- **GVHD prophylaxis regimens**: prevention strategies affect survival trade-offs.  
- **ABO donor–recipient match**: modest but reproducible effect.  
- **Comorbidity/performance scores**: weaker but meaningful contributions.  

**Visual Placeholder:**  
![Feature Importance Placeholder](visuals/classification/svm/svm_feat_importance.png)  

---

## Clinical Impact
- **Reinforces known drivers**: Age and remission status dominate risk.  
- **Highlights modifiable factors**: Strategies for GVHD prophylaxis matter.  
- **Suggests subtle donor effects**: ABO compatibility plays a role.  
- **Proof-of-concept**: Even modest ML models recover clinically interpretable signals, supporting eventual use for **decision-support tools** in transplant planning.  

---

## Limitations
- Survival modeled as **binary** → ignores censoring/time-to-event structure.  
- **Row deletion** for missing data reduces sample size and may bias results.  
- **Nominal categorical variables treated as ordinal** → may distort results.  
- No external validation performed yet.  

---

## Recommendations & Future Directions
1. Move to **survival and competing-risks models** (Cox PH, Fine-Gray, DeepSurv).  
2. Implement **multiple imputation** instead of row deletion.  
3. Apply **one-hot encoding** and engineer interaction terms.  
4. Evaluate **calibration** (Brier score, reliability plots).  
5. Add **SHAP-based explainability** for local/global interpretation.  
6. Perform **external/temporal validation**.  
7. Extend to **donor recommendation system**.  

---

## Next Steps (Checklist)
- [ ] Implement survival analysis module  
- [ ] Improve missing data handling  
- [ ] Add one-hot encoding  
- [ ] Assess calibration and explainability  
- [ ] Expand to donor recommendation framework  
