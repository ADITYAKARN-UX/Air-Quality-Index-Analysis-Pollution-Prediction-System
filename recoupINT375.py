import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\ADITYA KARN\OneDrive\Desktop\air quality index\realtimeairqualityindex.csv")

# Replace string 'NA' with proper NaN
df.replace('NA', np.nan, inplace=True)
df['latitude']      = pd.to_numeric(df['latitude'],      errors='coerce')
df['longitude']     = pd.to_numeric(df['longitude'],     errors='coerce')
df['pollutant_min'] = pd.to_numeric(df['pollutant_min'], errors='coerce')
df['pollutant_max'] = pd.to_numeric(df['pollutant_max'], errors='coerce')
df['pollutant_avg'] = pd.to_numeric(df['pollutant_avg'], errors='coerce')

# ==============================================================================
#                           BASIC EXPLORATION
# ==============================================================================
print(df.info())
print("\nShape of dataset:", df.shape)
print(df.head())
print(df.tail())

print("\nUnique pollutants count:", df["pollutant_id"].nunique())
print("Unique pollutants:", df["pollutant_id"].unique())

# Q: Count occurrence in "pollutant_id"?
print("\nCount occurrence in pollutant_id:")
print(df["pollutant_id"].value_counts())
print(df.describe())

print("\nMean of pollutant_avg:",    df["pollutant_avg"].mean().round(1))
print("Median of pollutant_avg:",   int(df["pollutant_avg"].median()))
print("Mean of pollutant_min:",     df["pollutant_min"].mean().round(1))
print("Median of pollutant_min:",   int(df["pollutant_min"].median()))
print("Std of pollutant_max:",      df["pollutant_max"].std().round(1))

# Q: Create a new column — pollutant spread (max - min)
df["pollutant_spread"] = df["pollutant_max"] - df["pollutant_min"]
print("\nNew column 'pollutant_spread' (max - min):")
print(df[["pollutant_min", "pollutant_max", "pollutant_spread"]].head())

# Normalization
print("\nMaximum pollutant_avg:", df["pollutant_avg"].max())
print("Minimum pollutant_avg:", df["pollutant_avg"].min())
print("Maximum latitude:",      df["latitude"].max())
print("Minimum latitude:",      df["latitude"].min())

df["pollutant_avg_normalized"] = (df["pollutant_avg"] - df["pollutant_avg"].min()) / \
                                  (df["pollutant_avg"].max() - df["pollutant_avg"].min())
print("\nNormalized 'pollutant_avg' column:")
print(df[["pollutant_avg", "pollutant_avg_normalized"]].head(10))

# Filtering
filtered_data = df[df["pollutant_avg"] > 100]
print("\nFiltered data where pollutant_avg > 100:")
print(filtered_data.head())

filtered_data1 = df[df["pollutant_id"] == "PM2.5"]
print("\nFiltered data for PM2.5 only:")
print(filtered_data1.head())

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Top 10 states by records
print("\nTop 10 States by record count:")
print(df["state"].value_counts().head(10))

# ==============================================================================
#                           VISUALIZATIONS
# ==============================================================================

# ---- 1. HISTOGRAM — pollutant_avg ----
plt.figure(figsize=(9, 5))
data_hist = df["pollutant_avg"].dropna()
data_hist = data_hist[data_hist < data_hist.quantile(0.97)]   # remove extreme outliers for clarity
plt.hist(data_hist, bins=30, color='steelblue', edgecolor='black')
plt.axvline(data_hist.mean(),   color='red',    linestyle='--', linewidth=2, label=f"Mean: {data_hist.mean():.1f}")
plt.axvline(data_hist.median(), color='orange', linestyle='--', linewidth=2, label=f"Median: {data_hist.median():.1f}")
plt.xlabel("Pollutant Average Value")
plt.ylabel("Frequency")
plt.title("Histogram of Pollutant Average Values")
plt.legend()
plt.xticks(rotation=15)
plt.yticks(rotation=15)
plt.tight_layout()
plt.show()

# ---- 2. BAR CHART — Top 12 States by Record Count ----
plt.figure(figsize=(12, 5))
state_counts = df["state"].value_counts().head(12)
state_counts.plot(kind="bar", color='steelblue', edgecolor='black')
plt.xlabel("State")
plt.ylabel("Number of Records")
plt.title("Top 12 States by Number of Air Quality Records")
plt.xticks(rotation=40, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---- 3. BAR CHART — Average Pollutant Level per Pollutant Type ----
plt.figure(figsize=(9, 5))
poll_avg = df.groupby("pollutant_id")["pollutant_avg"].mean().sort_values(ascending=False)
poll_avg.plot(kind="bar", color='tomato', edgecolor='black')
plt.xlabel("Pollutant Type")
plt.ylabel("Mean Average Value (µg/m³ or ppb)")
plt.title("Average Pollution Level per Pollutant Type")
plt.xticks(rotation=20)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# ---- 4. HISTOGRAM — pollutant_min ----
plt.figure(figsize=(9, 5))
data_min = df["pollutant_min"].dropna()
data_min = data_min[data_min < data_min.quantile(0.97)]
plt.hist(data_min, bins=30, color='mediumseagreen', edgecolor='black')
plt.xlabel("Pollutant Minimum Value")
plt.ylabel("Frequency")
plt.title("Histogram of Pollutant Minimum Values")
plt.xticks(rotation=15)
plt.yticks(rotation=15)
plt.tight_layout()
plt.show()

# ---- 5. SCATTER PLOT — Latitude vs Pollutant Average ----
plt.figure(figsize=(9, 5))
colors_map = {'PM2.5':'red','PM10':'orange','NO2':'blue','SO2':'gold',
              'CO':'purple','NH3':'green','OZONE':'cyan'}
for poll in df["pollutant_id"].dropna().unique():
    sub = df[df["pollutant_id"] == poll].dropna(subset=["latitude","pollutant_avg"])
    plt.scatter(sub["latitude"], sub["pollutant_avg"],
                color=colors_map.get(poll, 'gray'), alpha=0.5, s=20, label=poll)
plt.xlabel("Latitude")
plt.ylabel("Pollutant Average Value")
plt.title("Scatter Plot: Latitude vs Pollutant Average")
plt.legend(fontsize=8, loc='upper right')
plt.xticks(rotation=15)
plt.yticks(rotation=15)
plt.tight_layout()
plt.show()

# ---- 6. SCATTER PLOT — Pollutant Min vs Max ----
plt.figure(figsize=(9, 5))
for poll in df["pollutant_id"].dropna().unique():
    sub = df[df["pollutant_id"] == poll].dropna(subset=["pollutant_min","pollutant_max"])
    plt.scatter(sub["pollutant_min"], sub["pollutant_max"],
                color=colors_map.get(poll, 'gray'), alpha=0.5, s=20, label=poll)
plt.xlabel("Pollutant Min")
plt.ylabel("Pollutant Max")
plt.title("Scatter Plot: Pollutant Min vs Max")
plt.legend(fontsize=8, loc='upper left')
plt.xticks(rotation=15)
plt.yticks(rotation=15)
plt.tight_layout()
plt.show()

# ---- 7. PIE CHART — Pollutant Type Distribution ----
plt.figure(figsize=(7, 7))
poll_counts = df["pollutant_id"].value_counts()
plt.pie(poll_counts.values, labels=poll_counts.index, autopct='%1.1f%%',
        colors=['steelblue','tomato','mediumseagreen','gold','purple','cyan','orange'],
        startangle=140, wedgeprops=dict(edgecolor='white', linewidth=1.5))
plt.title("Pie Chart: Pollutant Type Distribution")
plt.tight_layout()
plt.show()

# ---- 8. PIE CHART — Top 6 States Distribution ----
plt.figure(figsize=(7, 7))
top6 = df["state"].value_counts().head(6)
plt.pie(top6.values, labels=top6.index, autopct='%1.1f%%',
        colors=['steelblue','tomato','mediumseagreen','gold','purple','orange'],
        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1.5))
plt.title("Pie Chart: Top 6 States by Record Count")
plt.tight_layout()
plt.show()

# ---- 9. LINE PLOT — Pollutant Avg for First 30 Records ----
plt.figure(figsize=(10, 5))
plt.plot(df["pollutant_avg"].head(30), marker='o', linestyle='--', color='steelblue', linewidth=2, markersize=6)
plt.xlabel("Record Index")
plt.ylabel("Pollutant Average Value")
plt.title("Line Plot: Pollutant Avg for First 30 Records")
plt.xticks(rotation=15)
plt.yticks(rotation=15)
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# ---- 10. LINE PLOT — Mean Avg per Pollutant (Min / Avg / Max) ----
plt.figure(figsize=(10, 5))
stats = df.groupby("pollutant_id")[["pollutant_min","pollutant_avg","pollutant_max"]].mean()
plt.plot(stats.index, stats["pollutant_min"], marker='s', linestyle='--', color='steelblue',  linewidth=2, label='Mean Min')
plt.plot(stats.index, stats["pollutant_avg"], marker='o', linestyle='-',  color='tomato',     linewidth=2, label='Mean Avg')
plt.plot(stats.index, stats["pollutant_max"], marker='^', linestyle='--', color='mediumseagreen', linewidth=2, label='Mean Max')
plt.xlabel("Pollutant Type")
plt.ylabel("Mean Value")
plt.title("Line Plot: Mean Min / Avg / Max per Pollutant")
plt.legend()
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# ---- 11. DONUT CHART — Pollutant Type ----
plt.figure(figsize=(7, 7))
plt.pie(poll_counts.values, labels=poll_counts.index, autopct='%1.1f%%',
        colors=['steelblue','tomato','mediumseagreen','gold','purple','cyan','orange'],
        startangle=140, wedgeprops=dict(edgecolor='white', linewidth=1.5))
plt.gca().add_artist(plt.Circle((0, 0), 0.5, color='white'))
plt.title("Donut Chart: Pollutant Type Share")
plt.tight_layout()
plt.show()

# ---- 12. BOXPLOT — Pollutant Avg per Pollutant Type ----
plt.figure(figsize=(11, 6))
box_data  = [df[df["pollutant_id"]==p]["pollutant_avg"].dropna().clip(upper=300).values
             for p in poll_counts.index]
bp = plt.boxplot(box_data, patch_artist=True,
                 medianprops=dict(color='black', linewidth=2))
box_colors = ['steelblue','tomato','mediumseagreen','gold','purple','cyan','orange']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
plt.xticks(range(1, len(poll_counts)+1), poll_counts.index, rotation=15)
plt.xlabel("Pollutant Type")
plt.ylabel("Average Value (µg/m³ or ppb)")
plt.title("Boxplot: Outlier Detection per Pollutant")
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# ---- 13. HEATMAP — Correlation Matrix ----
plt.figure(figsize=(8, 5))
corr = df[["pollutant_min","pollutant_max","pollutant_avg","pollutant_spread","latitude","longitude"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.8, annot_kws={'size': 10})
plt.title("Correlation Heatmap")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# ---- 14. HORIZONTAL BAR — Top 15 Most Polluted Cities (PM2.5) ----
plt.figure(figsize=(10, 7))
pm25_city = (df[df["pollutant_id"]=="PM2.5"]
             .groupby("city")["pollutant_avg"]
             .mean()
             .sort_values(ascending=False)
             .head(15))
pm25_city[::-1].plot(kind='barh', color='tomato', edgecolor='black')
plt.xlabel("Mean PM2.5 Value (µg/m³)")
plt.ylabel("City")
plt.title("Top 15 Most Polluted Cities by PM2.5")
plt.axvline(60, color='black', linestyle='--', linewidth=1.5, label='Unsafe Limit ~60')
plt.legend()
plt.tight_layout()
plt.show()

# ---- 15. COUNT PLOT — Records per Pollutant ----
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="pollutant_id", order=poll_counts.index,
              hue="pollutant_id", 
              palette=['steelblue','tomato','mediumseagreen','gold','purple','cyan','orange'],
              edgecolor='black',
              legend=False 
              )
plt.xlabel("Pollutant Type")
plt.ylabel("Count")
plt.title("Count Plot: Number of Readings per Pollutant")
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# ---- using scikit-learn LabelEncoder ----
from sklearn.preprocessing import LabelEncoder
sample_states = ["Maharashtra", "Delhi", "Bihar", "Delhi", "Maharashtra"]
label_encoder = LabelEncoder()
encoded_states = label_encoder.fit_transform(sample_states)
print("\nOriginal data:", sample_states)
print("Encoded data:", encoded_states)


# ==============================================================================
#                        MACHINE LEARNING SECTION
# ==============================================================================

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# -----------------------------------------------------------------------
# STEP 1 : PREPARE TARGET — classify pollutant_avg into High / Low
# -----------------------------------------------------------------------
# High pollution = avg > median, Low = avg <= median
df_ml = df[["state","city","pollutant_id","pollutant_min",
            "pollutant_max","pollutant_avg","latitude","longitude"]].dropna().copy()

median_val = df_ml["pollutant_avg"].median()
df_ml["pollution_level"] = (df_ml["pollutant_avg"] > median_val).astype(int)
# 1 = High pollution, 0 = Low pollution

print("\nTarget value counts (1=High, 0=Low):")
print(df_ml["pollution_level"].value_counts())

# -----------------------------------------------------------------------
# STEP 2 : ENCODE CATEGORICAL COLUMNS
# -----------------------------------------------------------------------
le = LabelEncoder()
for col in ["state", "city", "pollutant_id"]:
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))

print("\nData after encoding (first 5 rows):")
print(df_ml.head())

# -----------------------------------------------------------------------
# STEP 3 : DEFINE FEATURES (X) AND TARGET (y)
# -----------------------------------------------------------------------
X = df_ml.drop(columns=["pollution_level", "pollutant_avg"])
y = df_ml["pollution_level"]

print("\nFeature shape:", X.shape)
print("Target shape: ", y.shape)
print("Target value counts:\n", y.value_counts())

# -----------------------------------------------------------------------
# STEP 4 : SPLIT DATA INTO TRAIN AND TEST
# -----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("\nTraining samples:", X_train.shape[0])
print("Testing  samples:", X_test.shape[0])

# -----------------------------------------------------------------------
# STEP 5 : LOGISTIC REGRESSION
# -----------------------------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

print("\n--- Logistic Regression ---")
print("Accuracy:", round(accuracy_score(y_test, lr_pred), 4))
print("Classification Report:\n", classification_report(y_test, lr_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, lr_pred), annot=True, fmt='d',
            cmap='Blues', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 6 : DECISION TREE CLASSIFIER
# -----------------------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("\n--- Decision Tree ---")
print("Accuracy:", round(accuracy_score(y_test, dt_pred), 4))
print("Classification Report:\n", classification_report(y_test, dt_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt='d',
            cmap='Oranges', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 7 : RANDOM FOREST CLASSIFIER
# -----------------------------------------------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\n--- Random Forest ---")
print("Accuracy:", round(accuracy_score(y_test, rf_pred), 4))
print("Classification Report:\n", classification_report(y_test, rf_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d',
            cmap='Greens', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 8 : K-NEAREST NEIGHBORS (KNN)
# -----------------------------------------------------------------------
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)

print("\n--- KNN (k=5) ---")
print("Accuracy:", round(accuracy_score(y_test, knn_pred), 4))
print("Classification Report:\n", classification_report(y_test, knn_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt='d',
            cmap='Purples', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title("Confusion Matrix - KNN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 9 : SUPPORT VECTOR MACHINE (SVM)
# -----------------------------------------------------------------------
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

print("\n--- SVM ---")
print("Accuracy:", round(accuracy_score(y_test, svm_pred), 4))
print("Classification Report:\n", classification_report(y_test, svm_pred))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, svm_pred), annot=True, fmt='d',
            cmap='Reds', xticklabels=['Low', 'High'],
            yticklabels=['Low', 'High'])
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 10 : MODEL COMPARISON BAR CHART
# -----------------------------------------------------------------------
model_names = ['Logistic Reg', 'Decision Tree', 'Random Forest', 'KNN', 'SVM']
accuracies = [
    accuracy_score(y_test, lr_pred),
    accuracy_score(y_test, dt_pred),
    accuracy_score(y_test, rf_pred),
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, svm_pred),
]

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, accuracies,
               color=['steelblue','orange','mediumseagreen','purple','tomato'],
               edgecolor='black')
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.ylim(0, 1.1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=15)
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

best_model = model_names[accuracies.index(max(accuracies))]
print("\nBest Model:", best_model, "with accuracy:", round(max(accuracies), 4))

# -----------------------------------------------------------------------
# STEP 11 : FEATURE IMPORTANCE (Random Forest)
# -----------------------------------------------------------------------
feat_imp = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
feat_imp.plot(kind='bar', color='mediumseagreen', edgecolor='black')
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.title("Feature Importances - Random Forest")
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 12 : ROC CURVE (all models)
# -----------------------------------------------------------------------
plt.figure(figsize=(8, 6))

for name, model, color in zip(
    ['Logistic Reg', 'Random Forest', 'KNN', 'SVM'],
    [lr_model,        rf_model,        knn_model, svm_model],
    ['steelblue',    'mediumseagreen', 'purple',  'tomato']
):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, linewidth=2,
             label=f'{name}  (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - All Models")
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# STEP 13 : PREDICT ON A SINGLE NEW RECORD
# -----------------------------------------------------------------------
print("\n--- Predicting on a single new record ---")
single_record = X_test.iloc[[0]]
actual_label  = y_test.iloc[0]

lr_single  = lr_model.predict(single_record)[0]
rf_single  = rf_model.predict(single_record)[0]
dt_single  = dt_model.predict(single_record)[0]

label_map = {1: "High Pollution", 0: "Low Pollution"}
print("Actual Pollution Level:", label_map[actual_label])
print("Logistic Regression   :", label_map[lr_single])
print("Random Forest         :", label_map[rf_single])
print("Decision Tree         :", label_map[dt_single])

print("\n--- Prediction Probability (Logistic Regression) ---")
prob = lr_model.predict_proba(single_record)[0]
print(f"Probability of Low  Pollution: {prob[0]:.4f}")
print(f"Probability of High Pollution: {prob[1]:.4f}")

# -----------------------------------------------------------------------
# STEP 14 : CORRELATION HEATMAP (after encoding)
# -----------------------------------------------------------------------
plt.figure(figsize=(9, 6))
corr_ml = df_ml.corr()
sns.heatmap(corr_ml, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=0.5, annot_kws={'size': 8})
plt.title("Correlation Heatmap - All ML Features")
plt.xticks(rotation=35, ha='right')
plt.tight_layout()
plt.show()

print("\n==================== DONE ====================")
