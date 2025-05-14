Communities and Crime Analysis Project
----------------------------------------------
This project analyzes socio-economic factors contributing to violent crimes using various machine learning models and statistical techniques. The dataset is derived from the UCI Communities and Crime dataset.

Structure ----
communitiesAndCrimeProject/
│
├── .venv/                          → Python virtual environment.
├── catboost_info/                 → CatBoost metadata folder (can be ignored).
├── communities+and+crime/         → Raw data files and processing scripts.
│   ├── communities.data           → Raw data file.
│   ├── communities.names          → Feature descriptions.
│   ├── comparator.py              → Model comparator script.
│   ├── correlation_rates.xlsx     → Correlation of features with crime rate.
│   ├── correlation_to_column100.xlsx → Specific correlation analysis.
│   ├── kinda_cleaned.xlsx         → Cleaned version of the dataset (manual process).
│   └── numbered_output.txt        → Output logs with numbered lines (for tracking).
│
├── data/                          → Processed data and generated outputs.
│   ├── cleaned_notebook.ipynb     → Main analysis notebook.
│   ├── communities_percentages_cleaned.csv → Cleaned CSV used for modeling.
│   ├── decision_tree_train_vs_cv.png → Accuracy visualization: Decision Tree.
│   ├── logistic_train_vs_cv.png   → Accuracy visualization: Logistic Regression.
│   ├── svm_train_vs_cv.png        → Accuracy visualization: SVM.
│   └── main.py                    → Alternate script for model training/testing.

How to Run the Project
----------------------------
Requirements ----
Make sure you have Python 3.8+ and the following libraries installed:
-os.
-numpy.
-pandas.
-matplotlib.
-seaborn.
-warnings.
-collections.
-sklearn.
-tensorflow.
-imblearn.
-networkx.
-catboost.

Running the notebook
----------------------------
1. Open cleaned_notebook.ipynb using Jupyter Notebook or VSCode with the Jupyter extension.
2. Run all cells to:
-Load and clean the dataset.
-Perform correlation analysis.
-Train multiple ML models (Logistic, Decision Tree, Random Forest, ANN, etc.)
-Evaluate and visualize performance.

Notebook Overview
----------------------------
The notebook includes:
-Data loading and preprocessing with pandas, SimpleImputer, and StandardScaler
-ML models like:
*Logistic Regression.
*Decision Tree.
*Random Forest.
*CatBoost.
*ANN with TensorFlow.
-SMOTE for handling imbalanced classes.
-Visualizations for comparing training and validation scores.
-External Excel/CSV integrations for exploratory analysis.
