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
st.write(df.describe())
st.write(df.info()
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

df_analysis = df.copy()

# Drop index column if exists
if 'index' in df_analysis.columns:
    df_analysis.drop(columns=['index'], inplace=True)

# Impute missing 'agent' values with median
df_analysis['agent'].fillna(df_analysis['agent'].median(), inplace=True)

st.subheader("Dataset Info")
st.text(df_analysis.info())

# ===============================
# Numeric Visualization Function
# ===============================
def num_visualization(df_analysis, col):
    fig_box = px.box(df_analysis, y=col, title=f"Boxplot of {col}")
    st.plotly_chart(fig_box, use_container_width=True)
    
    fig_hist = px.histogram(df_analysis, x=col, nbins=50, title=f"Histogram of {col}")
    st.plotly_chart(fig_hist, use_container_width=True)

# Loop over numeric columns
num_cols = df_analysis.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in num_cols:
    num_visualization(df_analysis, col)
    st.write("-" * 50)
    st.write(df_analysis[col].describe())

# ===============================
# Categorical Column Summaries
# ===============================
cat_cols = df_analysis.select_dtypes(include=["object"]).columns.tolist()
for col in cat_cols:
    st.write("-" * 50)
    st.write(col)
    st.write(df_analysis[col].value_counts(normalize=True))

# ===============================
# Pie Charts for Selected Categorical Columns
# ===============================

# Top 10 cities
vc = df_analysis['city'].value_counts(normalize=True)
top_10_cities = vc.head(10)
fig_cities = px.pie(data_frame=top_10_cities, names=top_10_cities.index, values=top_10_cities.values,
                    title='Top 10 Cities by Proportion of Bookings')
st.plotly_chart(fig_cities, use_container_width=True)

# Customer type
vc = df_analysis['customer_type'].value_counts(normalize=True)
top_2 = vc.head(2)
others_sum = vc.iloc[2:].sum()
top_2 = top_2.append(pd.Series([others_sum], index=['Other']))
fig_customer = px.pie(data_frame=top_2, names=top_2.index, values=top_2.values, title='Customer Types Distribution')
st.plotly_chart(fig_customer, use_container_width=True)

# Market segment
fig_market = px.pie(df_analysis, names='market_segment', title='Distribution of Market Segment Types')
st.plotly_chart(fig_market, use_container_width=True)

# Meal types
fig_meal = px.pie(df_analysis, names='meal', title='Proportion of Meal Types')
st.plotly_chart(fig_meal, use_container_width=True)

# Top 10 countries
vc = df_analysis['country'].value_counts(normalize=True)
top_10_countries = vc.head(9)
others_sum = vc.iloc[9:].sum()
top_10_countries = top_10_countries.append(pd.Series([others_sum], index=['Other']))
fig_country = px.pie(data_frame=top_10_countries, names=top_10_countries.index, values=top_10_countries.values,
                     title='Top 10 Countries by Proportion of Bookings')
st.plotly_chart(fig_country, use_container_width=True)

# Cancellations
fig_cancellation = px.pie(df_analysis, names='is_canceled',
                          title='Distribution of Cancellations',
                          color='is_canceled',
                          color_discrete_map={0: 'green', 1: 'red'})
st.plotly_chart(fig_cancellation, use_container_width=True)


# ===============================
# Bivariate Analysis
# ===============================
st.subheader("Bivariate Analysis")

for col in num_cols:
    if col != 'is_canceled':
        fig = px.box(df, x='is_canceled', y=col, title=f"{col} vs Cancellation")
        st.plotly_chart(fig, use_container_width=True)

def cross_tabulation(df, col1, col2="is_canceled"):
    ct = pd.crosstab(df[col1], df[col2], normalize='index') * 100
    st.write(f"Cross-tabulation of {col1} vs {col2}")
    st.dataframe(ct.round(2))

cat_cols = [col for col in df_analysis.columns if df_analysis[col].dtype == 'object' and col != 'is_canceled']
for col in cat_cols:
    cross_tabulation(df_analysis, col)

for col in cat_cols:
    fig = px.histogram(df_analysis, x=col, color='is_canceled',
                       barmode='group',
                       title=f'Cancellations by {col} (Grouped)')
    st.plotly_chart(fig, use_container_width=True)
from scipy.stats import ttest_ind

st.subheader("T-tests for Numeric Features")

for col in num_cols:
    col_group1 = df_analysis[df_analysis['is_canceled'] == 1][col]
    col_group2 = df_analysis[df_analysis['is_canceled'] == 0][col]

    t_stat, p_value = ttest_ind(
        col_group1,
        col_group2,
        equal_var=False,
        nan_policy='omit'
    )

    st.write(f"**{col}**")
    st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
    st.write("---")
from scipy.stats import chi2_contingency

st.subheader("Chi-square Test for Categorical Features")

for col in cat_cols:
    table = pd.crosstab(df_analysis[col], df_analysis['is_canceled'])
    chi2, p_value, dof, expected = chi2_contingency(table)

    st.write(f"**{col}**")
    st.write(f"Chi-square: {chi2:.4f}, P-value: {p_value:.4f}, Degrees of Freedom: {dof}")
    st.write("---")
# ===============================
# Correlation Heatmap
# ===============================
st.subheader("Correlation Matrix")

corr = df.select_dtypes(include=['int64', 'float64']).corr()

fig, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0, ax=ax)
st.pyplot(fig)
num_cols = [
    'lead_time',
    'required_car_parking_spaces',
    'total_of_special_requests',
    'is_canceled'
]

df_pair = df_analysis[num_cols]

st.subheader("Pairplot of Selected Numeric Features")
pair_fig = sns.pairplot(df_pair, hue='is_canceled', diag_kind='kde')
st.pyplot(pair_fig.fig)  # Use .fig to get the figure object

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







