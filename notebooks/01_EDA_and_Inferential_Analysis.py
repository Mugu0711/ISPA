import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from utils.utils import plot_hist_box, plot_bar_percent

# Load dataset
data_path = "../data/Telco-Customer-Churn.csv"
df = pd.read_csv(data_path)

# ----------------------- BASIC INFO -----------------------
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

# Remove CustomerID
df.drop(columns=['customerID'], inplace=True)

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ----------------------- EDA -----------------------
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_cols = [c for c in df.columns if c not in num_cols + ['Churn']]

for col in num_cols:
    plot_hist_box(df, col)

for col in cat_cols[:5]:
    plot_bar_percent(df, col, 'Churn')

# ----------------------- INFERENTIAL STATISTICS -----------------------
anova_results = []
for col in num_cols:
    churned = df[df['Churn'] == 1][col]
    retained = df[df['Churn'] == 0][col]
    f_stat, p_val = stats.f_oneway(churned, retained)
    anova_results.append([col, f_stat, p_val])

anova_df = pd.DataFrame(anova_results, columns=['Variable', 'F-Statistic', 'p-Value'])
print("\nANOVA Results:\n", anova_df)

chi_results = []
for col in cat_cols:
    table = pd.crosstab(df[col], df['Churn'])
    chi2, p, dof, ex = stats.chi2_contingency(table)
    chi_results.append([col, chi2, p])

chi_df = pd.DataFrame(chi_results, columns=['Variable', 'Chi2', 'p-Value']).sort_values('p-Value')
print("\nChi-Square Results:\n", chi_df.head(10))

# ----------------------- PLOTS -----------------------
plt.figure(figsize=(8, 4))
sns.barplot(data=anova_df, x='Variable', y=-np.log10(anova_df['p-Value']), color='teal')
plt.title("ANOVA Significance (-log10 p-value)")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(data=chi_df.head(12), x=-np.log10(chi_df['p-Value'].head(12)), y='Variable', color='orange')
plt.title("Top 12 Categorical Predictors by Chi-Square Significance")
plt.tight_layout()
plt.show()
