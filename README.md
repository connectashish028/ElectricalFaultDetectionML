# Fault Detection in Power System Transmission Lines

This project focuses on enhancing fault detection and classification in power system transmission lines using machine learning models.

## Table of Contents

1. **Introduction:**
   - Project objectives and motivation.
   - Importance of fault detection in power systems.

2. **Dataset Description:**
   - Data source and overview.
   - Explanation of features and fault types.

3. **Data Preprocessing:**
   - Handling missing values (if any).
   - Feature engineering and selection.
   - Data normalization/scaling.

4. **Exploratory Data Analysis (EDA):**
   - Visualizing fault patterns and distributions.
   - Analyzing feature correlations.

5. **Machine Learning Models:**
   - Overview of selected models (ANN, SVM, Decision Tree, Random Forest, XGBoost, LSTM).
   - Model implementation and training.
   - Hyperparameter tuning and optimization.

6. **Fault Detection and Classification:**
   - Evaluation of model performance on test data.
   - Performance metrics (accuracy, precision, recall, F1-score).
   - Model comparison and selection.

7. **Data Visualization:**
   - Visualizing fault detection results (e.g., confusion matrices).
   - Comparing model performance using appropriate plots.

8. **Results and Discussion:**
   - Summary of key findings and insights.
   - Implications for fault detection systems.
   - Limitations and potential areas for improvement.

9. **Conclusion:**
   - Main conclusions and takeaways.
   - Recommendations for practical implementation.
   - Future research directions.

10. **References:**
    - List of relevant literature and resources.

11. **Appendices (if applicable):**
    - Additional data or code.
    - Model specifications and configurations.


## Workflow

The project follows a typical machine learning workflow for fault detection:

1. **Data Acquisition & Preprocessing:** Gather the dataset containing information about transmission line parameters and fault occurrences. Clean the data, handle missing values, and prepare it for analysis.

2. **Exploratory Data Analysis (EDA):** Analyze the data to understand the relationships between different features and identify potential patterns associated with faults. Visualizations like graphs and plots are used to gain insights.

3. **Feature Engineering & Selection:** Extract relevant features from the dataset or create new ones that might improve the accuracy of fault detection. Select the most informative features for model training.

4. **Model Selection & Training:** Choose suitable machine learning models based on the nature of the data and the specific fault detection task. Train the selected models using the prepared dataset.

5. **Model Evaluation & Comparison:** Evaluate the performance of different models using appropriate metrics such as accuracy, precision, recall, and F1-score. Compare the performance of different models to identify the most effective one.

6. **Visualization & Interpretation:** Visualize the results of the fault detection process using techniques like confusion matrices and other relevant plots. Interpret the results to understand the strengths and weaknesses of the models.

7. **Deployment & Monitoring:** Deploy the chosen model for real-time or offline fault detection in a power system. Continuously monitor the model's performance and retrain or adjust it as needed.


## Project Description

Power system transmission lines are critical infrastructure for delivering electricity, and ensuring their reliable operation is paramount. This project aims to develop and evaluate machine learning models for accurately detecting and classifying faults in transmission lines.

## Features

- **Data Analysis:** Comprehensive exploratory data analysis (EDA) to understand patterns and relationships in the dataset.
- **Fault Visualization:** Visualizations to illustrate fault patterns and characteristics.
- **Machine Learning Models:** Implementation and evaluation of various machine learning models, including:
    - Artificial Neural Networks (ANNs)
    - Support Vector Machines (SVMs)
    - Decision Trees
    - Random Forests
    - XGBoost
    - LSTM
- **Model Comparison:** Comparative analysis of model performance based on relevant metrics.
- **Result Visualization:** Clear and informative visualizations of fault detection results and model performance.

## Installation

1. Clone the repository: `git clone <repository_url>`
2. Install required libraries: `pip install -r requirements.txt`

## Usage

1. Download the dataset (`classData.csv`)
2. Place the dataset in the `data` directory.
3. Run the Jupyter notebook (`Fault_Detection.ipynb`) to perform data analysis, model training, and evaluation.

## Dataset Details

1. **Source:** Kaggle Datasets
2. **Description:** The dataset contains simulated data of a power system transmission line under various fault conditions.
3. **Features:**
    * `G, C, B, A`: Fault indicators (0 or 1) for ground, line C, line B, and line A, respectively.
    * `Ia, Ib, Ic`: Current measurements for phases A, B, and C.
    * `Va, Vb, Vc`: Voltage measurements for phases A, B, and C.
  
## Model Performance Summary
| Model | CorrR² | MSE | RMSE | MAE | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.23 | -1.016 | 6.402 | 2.531 | 0.36 | 0.34 | 0.36 | 0.24 |
| Decision Tree | 0.98 | 0.97 | 0.11 | 0.32 | 0.10 | 0.90 | 0.90 | 0.90 |
| Random Forest | 0.98 | 0.96 | 0.12 | 0.34 | 0.11 | 0.89 | 0.89 | 0.89 |
| Support Vector Machine | 0.89 | 0.77 | 0.74 | 0.86 | 0.36 | 0.76 | 0.73 | 0.76 |
| XGBClassifier | 0.97 | 0.94 | 0.18 | 0.43 | 0.17 | 0.83 | 0.83 | 0.83 |
| LSTM | 0.99500 | 0.9900 | 0.0195 | 0.1396 | 0.0508 | N/A | N/A | N/A |

## Results
* The models achieved high accuracy in detecting and classifying faults.
* XGBoost, Random Forest, and the LSTM model showed superior performance compared to other models.
* Detailed results and visualizations are available in the Jupyter notebook.

## Future Work
* Explore more sophisticated deep learning models.
* Implement real-time fault detection and analysis.
* Develop a user-friendly interface for visualizing and interpreting results.
