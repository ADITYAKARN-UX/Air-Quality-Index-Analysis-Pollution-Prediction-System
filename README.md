🌍 Air Quality Index Analysis & Pollution Prediction System
📌 Overview

This project focuses on Air Quality Index (AQI) analysis and pollution prediction using real-world environmental data. It performs data cleaning, exploratory data analysis (EDA), visualization, and machine learning classification to determine pollution levels.

The system analyzes pollutants like PM2.5, PM10, NO₂, SO₂, CO, NH₃, and Ozone, and predicts whether pollution levels are High or Low based on environmental features.

📂 Dataset
File:
Contains:
City, State
Pollutant Type
Pollutant Min, Max, Avg
Latitude & Longitude
Real-world AQI observations across multiple locations.
⚙️ Features
🔹 Data Preprocessing
Handles missing values (NA → NaN)
Converts columns into numeric format
Feature engineering:
pollutant_spread = max - min
Normalization of pollutant values
🔹 Exploratory Data Analysis (EDA)
Dataset shape, structure, statistics
Pollutant distribution analysis
State-wise and pollutant-wise insights
Filtering high pollution regions
🔹 Data Visualization

Includes multiple graphs:

Histogram (pollutant distribution)
Bar charts (state & pollutant comparison)
Scatter plots (geographical impact)
Pie & Donut charts (distribution)
Line plots (trend analysis)
Boxplots (outlier detection)
Heatmaps (correlation analysis)
🔹 Machine Learning Models

The project compares multiple classification models:

Model	Purpose
Logistic Regression	Baseline model
Decision Tree	Rule-based classification
Random Forest	Ensemble learning
K-Nearest Neighbors	Distance-based
Support Vector Machine	Advanced classification
🧠 ML Objective
Convert pollution data into classification:
1 → High Pollution
0 → Low Pollution
Based on median pollutant value
📊 Model Evaluation
Accuracy Score
Confusion Matrix
Classification Report
ROC Curve & AUC Score
Feature Importance (Random Forest)
📈 Output Highlights
Best performing model is selected automatically
Pollution prediction for new data
Probability estimation for predictions
🛠️ Tech Stack
Python
Libraries Used:
pandas
numpy
matplotlib
seaborn
scikit-learn
🚀 How to Run
# Step 1: Clone the repository
git clone https://github.com/your-username/air-quality-analysis.git

# Step 2: Navigate into project folder
cd air-quality-analysis

# Step 3: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Step 4: Run the script
python your_script_name.py
📌 Key Insights
PM2.5 is one of the most critical pollutants
Pollution varies significantly by location
Strong correlation exists between pollutant features
Machine learning can effectively classify pollution levels
🔮 Future Improvements
Real-time AQI integration via API
Deployment as web dashboard
Severity Index (custom metric)
Time-series forecasting
Integration with weather prediction system
🤝 Contribution

Contributions are welcome! Feel free to fork this repo and improve the project.

📜 License

This project is open-source and available under the MIT License.
