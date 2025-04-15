import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv("D:/Python/scikit-learn/train.csv")
print(df.shape)
df.head()

# Check null values
df.isnull().sum().sort_values(ascending=False).head(10)

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", annot=False)
plt.title("Feature Correlation")
plt.show()

# Drop columns with too many missing values or IDs
df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], axis=1)

# Fill remaining NaNs
df = df.fillna(df.median(numeric_only=True))

# Convert categoricals using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict
y_pred = model.predict(X_test_scaled)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE:", rmse)