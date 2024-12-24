# DiseaseClassifications

This repository contains untrained machine learning models developed as part of my Bachelor's thesis. These models aim to classify diseases based on various datasets, exploring the use of non-invasive features for prediction.

## Repository Structure

Each folder corresponds to a specific disease and contains one or two models:

- **1 Model**: For diseases where the folder contains only one model, it was trained using only **non-invasive features**.
- **2 Models**: For diseases where two models are present:
  - One model was trained on **all features** available in the dataset.
  - The other model was trained using **only non-invasive features**.

### Diseases and Models

The following diseases are included in this repository:

- Alzheimer
- Breast Cancer
- Chronic Kidney Disease
- Depression
- Differentiated Thyroid Cancer Recurrence
- Heart Disease
- Multiple Sclerosis
- Pathological Voice


# DiseaseClassifications

This repository contains untrained machine learning models developed as part of my Bachelor's thesis. These models aim to classify diseases based on various datasets, exploring the use of non-invasive features for prediction.

## Repository Structure

Each folder corresponds to a specific disease and contains one or two models:

- **1 Model**: For diseases where the folder contains only one model, it was trained using only **non-invasive features**.
- **2 Models**: For diseases where two models are present:
  - One model was trained on **all features** available in the dataset.
  - The other model was trained using **only non-invasive features**.

### Diseases and Models

The following diseases are included in this repository:

- Alzheimer
- Breast Cancer
- Chronic Kidney Disease
- Depression
- Differentiated Thyroid Cancer Recurrence
- Heart Disease
- Multiple Sclerosis
- Pathological Voice

## Metrics

### Heart Disease

**Metrics of Non-Invasive Model on Unseen Test Data**

| Model       | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|-------------|---------|-------------|-------------|----------|----------|
| NeuralNet   | 0.945   | 0.915       | 0.909       | 0.931    | 0.912    |

**Metrics of AF (All Features) Model on Unseen Test Data**

| Model       | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|-------------|---------|-------------|-------------|----------|----------|
| NeuralNet   | 0.927   | 0.898       | 0.818       | 0.898    | 0.869    |

### Alzheimer

**Metrics of Non-Invasive Model on Unseen Test Data**

| Metric       | Value   |
|--------------|---------|
| Test Accuracy| 0.8419  |
| Test AUC     | 0.8455  |

**Metrics of AF (All Features) Model on Unseen Test Data**

| Metric       | Value   |
|--------------|---------|
| Test Accuracy| 0.8093  |
| Test AUC     | 0.8117  |

### Breast Cancer

**Metrics of Non-Invasive Model on Unseen Test Data**

| Model       | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|-------------|---------|-------------|-------------|----------|----------|
| NeuralNet   | 0.918   | 0.857       | 0.857       | 0.799    | 0.857    |

**Metrics of AF (All Features) Model on Unseen Test Data**

| Model       | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|-------------|---------|-------------|-------------|----------|----------|
| NeuralNet   | 0.928   | 0.857       | 0.928       | 0.857    | 0.904    |


### Depression

**Metrics of Non-Invasive Model on Unseen Test Data**

| Metric       | Value   |
|--------------|---------|
| Test Accuracy| 0.9779  |



### Chronic Kidney Disease  
**Metrics of Non-Invasive Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 1.000   | 1.000       | 1.000       | 1.000    | 1.000    |
 
**Metrics of AF (All Features) Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 0.964   | 0.928       | 1.000       | 0.962    | 0.950    |


### Multiple Sclerosis  
**Metrics of Non-Invasive Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 0.833   | 0.761       | 0.750       | 0.761    | 0.756    |

**Metrics of AF (All Features) Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 0.830   | 0.761       | 0.750       | 0.761    | 0.756    |


### Pathological Voice  
**Metrics of Non-Invasive Model on Unseen Test Data**

| Model                | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|----------------------|---------|-------------|-------------|----------|----------|
| Random Forest        | 0.677   | 0.888       | 0.466       | 0.813    | 0.738    |
| Random Forest Top 18 | 0.859   | 0.851       | 0.866       | 0.884    | 0.857    |


### Differentiated Thyroid Cancer Recurrence  
**Metrics of Non-Invasive Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 0.898   | 0.900       | 0.928       | 0.857    | 0.921    |

**Metrics of AF (All Features) Model on Unseen Test Data**

| Model      | ROC-AUC | Sensitivity | Specificity | F1-Score | Accuracy |
|------------|---------|-------------|-------------|----------|----------|
| NeuralNet  | 1.000   | 1.000       | 1.000       | 1.000    | 1.000    |




## Note on Weights

The models provided in this repository are **untrained**. If you require the trained weights for these models, please contact me directly.

## Authors

- Iraj Masoudian



## Note on Weights

The models provided in this repository are **untrained**. If you require the trained weights for these models, please contact me directly.

## Authors

- Iraj Masoudian
