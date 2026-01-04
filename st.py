# ===============================
# Imports
# ===============================
import pandas as pd
import numpy as np
import streamlit as st
import plotly as px 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ===============================
# App Title
# ===============================
st.title("Hotel Cancellation Prediction ML App")

# ===============================
# Load Data
# ===============================
url = "https://raw.githubusercontent.com/bloxxastro1/Hotel_ML_Project/main/hotel_bookings_updated_2024.csv"
df = pd.read_csv(url)

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# ===============================
# Data Cleaning
# ===============================
# Drop leakage columns
leakage_cols = [
    'reservation_status', 'booking_changes', 'previous_cancellations',
    'deposit_type', 'days_in_waiting_list', 'arrival_date_month',
    'arrival_date_week_number', 'arrival_date_day_of_month'
]
df.drop(columns=leakage_cols, inplace=True)

# Handle missing values
df.dropna(subset=['children', 'country'], inplace=True)
df['market_segment'].replace("Undefined", np.nan, inplace=True)
df.dropna(subset=['market_segment'], inplace=True)

# Fix data types
df['children'] = df['children'].astype(int)
for col in df.columns:
    if df[col].nunique() == 2:
        df[col] = df[col].astype('object')

# Log-transform numeric columns safely
num_cols_log = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if 'is_canceled' in num_cols_log:
    num_cols_log.remove('is_canceled')  # do NOT transform target
for col in num_cols_log:
    df[col] = np.log1p(df[col])

# Create hotel type
df['Hotel_Type'] = df['hotel'].apply(lambda x: "Resort Hotel" if "Resort" in x else "City Hotel")
df.drop(columns=['hotel'], inplace=True)

# Filter out negative ADR
df = df[df['adr'] >= 0]

# Drop company column (too many missing)
df.drop(columns=['company'], inplace=True)

# Total nights
df['stays_total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
df.drop(columns=['stays_in_week_nights', 'stays_in_weekend_nights'], inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

st.subheader("Cleaned Data Preview")
st.dataframe(df.head())
# ===============================
# Univariate Analysis
# ===============================
st.subheader("Univariate Analysis")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in num_cols:
    fig = plt.histogram(df, x=col, title=f"Distribution of {col}")
    st.plotly_chart(fig, use_container_width=True)

# ===============================
# Bivariate Analysis
# ===============================
st.subheader("Bivariate Analysis")

for col in num_cols:
    if col != 'is_canceled':
        fig = px.box(df, x='is_canceled', y=col, title=f"{col} vs Cancellation")
        st.plotly_chart(fig, use_container_width=True)

# ===============================
# Correlation Heatmap
# ===============================
st.subheader("Correlation Matrix")

corr = df.select_dtypes(include=['int64', 'float64']).corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)

# ===============================
# Train-Test Split
# ===============================
X = df.drop('is_canceled', axis=1)
y = df['is_canceled'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ===============================
# Handle Class Imbalance (Training Only)
# ===============================
train_df = pd.concat([X_train, y_train], axis=1)
df_minority = train_df[train_df.is_canceled == 1]
df_majority = train_df[train_df.is_canceled == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

train_balanced = pd.concat([df_majority, df_minority_upsampled])
X_train = train_balanced.drop(columns=['is_canceled'])
y_train = train_balanced['is_canceled'].astype(int)

st.write(f"Training set size after balancing: {X_train.shape[0]} samples")

# ===============================
# Preprocessing Pipeline
# ===============================
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Ensure categorical columns are strings
for col in cat_cols:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols)
])

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier(
        n_estimators=100,      # smaller for faster Streamlit
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ))
])

# ===============================
# Cross-Validation
# ===============================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
    return_train_score=True
)

st.subheader("Cross-Validation Results")
st.write(f"Train F1: {scores['train_f1_weighted'].mean():.4f}")
st.write(f"Test F1 : {scores['test_f1_weighted'].mean():.4f}")

# ===============================
# Fit Final Model
# ===============================
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# ===============================
# Evaluation
# ===============================
st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)

roc_auc = roc_auc_score(y_test, y_proba)
st.write(f"ROC-AUC Score: {roc_auc:.4f}")




