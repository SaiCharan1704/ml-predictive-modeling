# ml-predictive-modeling
# Predictive Modeling Using Machine Learning

> Build a model to predict outcomes based on given data.

## Overview
This project demonstrates supervised learning by training and evaluating three classic classification algorithms:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**

## Key Features
- Apply algorithms: Linear Regression, Decision Trees, Random Forest
- Train and test models for accuracy
- Visualize performance using confusion matrices and ROC curves

## Expected Outcome
Gain experience in supervised learning and model evaluation.

## Project Structure
```
ml_project/
├── model.py              # Main ML pipeline
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/ml-predictive-modeling.git
cd ml-predictive-modeling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the model
python model.py
```

## Output
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~85% |
| Decision Tree | ~88% |
| Random Forest | ~91% |

Generated plots:
- `confusion_matrices.png` — side-by-side confusion matrices for all three models
- `roc_curves.png` — ROC curves with AUC scores

## License
MIT
