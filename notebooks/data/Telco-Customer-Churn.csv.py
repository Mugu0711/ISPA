import os
import pandas as pd

# Auto-create data folder if missing
os.makedirs("D:/ISPA/data", exist_ok=True)

data_path = "D:/ISPA/data/Telco-Customer-Churn.csv"

# ✅ Step 1: Check if file exists
if not os.path.exists(data_path):
    print("⚠️ Dataset not found. Downloading automatically...")
    try:
        url = "https://raw.githubusercontent.com/ybifoundation/Dataset/main/Telco%20Customer%20Churn.csv"
        df = pd.read_csv(url)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset downloaded and saved to {data_path}")
    except Exception as e:
        print("❌ Download failed! Generating synthetic dataset instead...")
        import numpy as np
        np.random.seed(42)
        n = 7043
        df = pd.DataFrame({
            'customerID': [f'CUST{i:05d}' for i in range(1, n + 1)],
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.randint(0, 2, n),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n),
            'tenure': np.random.randint(1, 73, n),
            'PhoneService': np.random.choice(['Yes', 'No'], n),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.4, 0.4, 0.2]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n),
            'MonthlyCharges': np.round(np.random.uniform(20, 120, n), 2),
            'TotalCharges': np.round(np.random.uniform(100, 9000, n), 2),
            'Churn': np.random.choice(['Yes', 'No'], n, p=[0.27, 0.73])
        })
        df.to_csv(data_path, index=False)
        print(f"✅ Synthetic dataset created and saved to {data_path}")
else:
    print("✅ Dataset found locally:", data_path)

# ✅ Step 2: Load dataset into memory
df = pd.read_csv(data_path)
print("📊 Dataset Loaded Successfully!")
print(df.shape)
print(df.head(3))
