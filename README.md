# Kyphosis-Disease-Classification-
Kyphosis is a spinal disorder characterized by an abnormal outward curvature of the spine, resulting in a hunchback or rounded upper back appearance. While some degree of curvature is normal, it is diagnosed when the curvature angle exceeds a certain threshold (typically 50 degrees).Kyphosis disease classification using logistic regression Model.

This project builds a complete Machine Learning pipeline to analyze the Kyphosis dataset and predict whether the spinal deformity Kyphosis is present after corrective surgery in children.
The workflow includes:
Data loading & exploration
Statistical summary of age (months → years)
Data preprocessing & label encoding
EDA and visualizations
Train–test split
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Model performance evaluation & comparison
    Dataset Description
The dataset contains postoperative outcomes for Kyphosis in children.
Feature	Description
Kyphosis-	Target variable (Present/Absent → encoded as 1/0)
Age-Age of the child in months
Number-Number of vertebrae involved in operation

The goal is binary classification → Predict Kyphosis present or not.
 Data Preprocessing
Label Encoding

The Kyphosis feature is encoded using LabelEncoder:
0 = Absent
1 = Present

Missing Values
No missing values are present in the dataset.

Age Statistics (in years)
Age stats are computed in both:
Months
Converted years (Age / 12)

Exploratory Data Analysis (EDA)
 Correlation Heatmap
Shows correlations between numerical features (Age, Number, Start), excluding the target.

Pairplot
Displays distribution and relationships between features.

Class Distribution
A countplot shows how many patients had Kyphosis after surgery.

Train–Test Split
The dataset is split using:
test_size = 0.2
random_state = 42

Features (X): Age, Number, Start
Target (y): Kyphosis

Machine Learning Models
1 Logistic Regression
Used as a baseline linear model.
Outputs:Confusion matrix
Classification report (Precision, Recall, F1-score)

2 Decision Tree Classifier
A non-linear model capable of capturing complex patterns.
Outputs:Confusion matrix
Classification report
Feature importance ranking

3️ Random Forest Classifier
An ensemble model that improves performance and reduces overfitting.
Outputs:
Confusion matrix
Classification report
Random Forest usually performs the best among all three models.

Evaluation Metrics
For each model, the following metrics are provided:
Accuracy
Precision
Recall
F1-score
Support

Confusion Matrix visualization

These metrics help compare model performance and select the best classifier
