# IMPORT LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from jupyterthemes import jtplot
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

#  Apply theme
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False)

# LOAD DATA
df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/kyphosis.csv')

print("\n===== Dataset Loaded =====\n")
print(df.head())
print(df.tail())
print(df.info())
print(df.describe())


# AGE STATISTICS IN YEARS
print("\n===== Age Statistics (in years) =====")
mean_age_months = df['Age'].mean()
mean_age_years = mean_age_months / 12

print(f"Mean Age (months): {mean_age_months:.2f}")
print(f"Mean Age (years): {mean_age_years:.2f}")
print(f"Min Age (years): {df['Age'].min() / 12:.2f}")
print(f"Max Age (years): {df['Age'].max() / 12:.2f}")


# LABEL ENCODING
encoder = LabelEncoder()
df['Kyphosis'] = encoder.fit_transform(df['Kyphosis'])

print("\n===== Encoded Dataset =====")
print(df.head())

# Percentage of Kyphosis cases
kyphosis_percent = (df['Kyphosis'].sum() / len(df)) * 100
print(f"\nDisease present after operation: {kyphosis_percent:.2f} %")

# VISUALIZATION
# CORRELATION HEATMAP(without kyphosis)
plt.figure(figsize=(8, 6))
corr = df.corr().drop(index="Kyphosis", columns="Kyphosis")
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (Without Kyphosis)")
plt.show()


# PAIRPLOT (WITHOUT TARGET)
sns.pairplot(df.drop(columns=["Kyphosis"]))
plt.show()

# CLASS DISTRIBUTION
plt.figure(figsize=(6, 4))
sns.countplot(x='Kyphosis', data=df, palette=['#4C72B0', '#55A868'])

# Annotate each bar
for p in plt.gca().patches:
    plt.text(p.get_x() + p.get_width() / 2, p.get_height(),
             f'{p.get_height()}', ha='center', va='bottom')

plt.title("Kyphosis Class Distribution")
plt.xlabel("Kyphosis")
plt.ylabel("Count")
plt.show()

# TRAIN-TEST SPLIT
X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n===== Train-Test Split Shapes =====")
print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# LOGISTIC REGRESSION MODEL
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

y_pred_lr = log_reg.predict(x_test)

print("\n===== Logistic Regression Performance =====")
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print(classification_report(y_test, y_pred_lr))

# DECISION TREE MODEL
dt = DecisionTreeClassifier(random_state=42)
dt.fit(x_train, y_train)

y_pred_dt = dt.predict(x_test)

print("\n===== Decision Tree Performance =====")
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix - Decision Tree")
plt.show()

print(classification_report(y_test, y_pred_dt))

# Feature Importance
feat_imp = pd.DataFrame(dt.feature_importances_,
                        index=X.columns,
                        columns=['Importance']).sort_values('Importance', ascending=False)
print("\n===== Feature Importance (Decision Tree) =====")
print(feat_imp)

# RANDOM FOREST MODEL
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("\n===== Random Forest Performance =====")
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges")
plt.title("Confusion Matrix - Random Forest")
plt.show()

print(classification_report(y_test, y_pred_rf))
