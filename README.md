# Breast Cancer Survival Prediction

This repository contains a machine learning model to predict the survival status (Alive or Dead) of breast cancer patients based on clinical and demographic data. The model was built using the Breast Cancer dataset, and several machine learning techniques were applied, including Logistic Regression and Random Forest.

## Libraries Used

- `pandas`: Data manipulation and analysis.
- `numpy`: Numerical computing.
- `matplotlib` & `seaborn`: Data visualization.
- `imblearn`: SMOTE (Synthetic Minority Over-sampling Technique) for handling class imbalance.
- `scikit-learn`: For model training, evaluation, and tuning.
- `joblib`: For model serialization.

## Dataset

The dataset `BRCA.csv` contains clinical and demographic data for 341 breast cancer patients. The target variable is `Patient_Status`, which indicates whether the patient is "Dead" or "Alive". The dataset includes 16 columns and various features such as tumor stage, age, gender, and treatment details.

### Key Observations:

- The dataset contains missing values and duplicate records, which were cleaned during the preprocessing.
- The target variable (`Patient_Status`) is imbalanced, with a higher proportion of "Alive" cases.
- Several columns like `ER status` and `PR status` had no variability and were dropped.
  
## Data Cleaning

1. **Missing Values**: The missing values in the dataset were less than 5%, and hence, rows with missing values were dropped.
2. **Redundant Columns**: Columns such as `Patient_ID`, `Date_of_Surgery`, and `Date_of_Last_Visit` were dropped as they did not contribute to the model.
3. **Encoding Categorical Variables**: Categorical features like `Tumour_Stage`, `Histology`, and `Gender` were encoded into numerical values.

## EDA (Exploratory Data Analysis)

1. **Target Variable Distribution**: The target variable was found to be imbalanced (80% Alive, 20% Dead).
2. **Age Distribution**: Most patients were concentrated in the age range of 45-65 years.
3. **Tumor Stage and Surgery Type Distribution**: Both were visualized using pie charts.
4. **Correlation Matrix**: A heatmap was generated to observe the relationships between numerical features.

## Model Building

### Models Used:

1. **Logistic Regression**:
   - A binary classifier was trained to predict survival status.
   - The model performed with 68.75% accuracy, but showed bias towards the majority class ("Alive").
   
2. **Random Forest**:
   - A Random Forest Classifier was trained, and its performance was better than Logistic Regression.
   - Hyperparameter tuning was done using GridSearchCV to improve the model.

### Model Evaluation:

- The accuracy, confusion matrix, and classification report were used to evaluate both models.
- The Random Forest model showed better performance but still required further optimization for more reliable results.

## Hyperparameter Tuning

- A grid search was performed on the Random Forest model to find the best hyperparameters (`n_estimators`, `max_depth`, etc.).
- The best-performing model was serialized using `joblib` and saved as `tuned_random_forest.pkl`.

## Conclusion

- The Random Forest model provided the best results, but its performance is not reliable enough for deployment without further improvements.
- Logistic Regression, despite being interpretable, showed bias due to the imbalanced dataset.

## Model Deployment

The best-performing model (Random Forest) has been serialized using `joblib` and is available for deployment.

```bash
joblib.dump(best_rf_model, 'tuned_random_forest.pkl')
