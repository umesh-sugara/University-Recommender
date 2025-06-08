# setup.py (Version 2)
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import pickle

print("Starting comprehensive setup script for both models...")

# --- 1. Load and Clean Data (from Notebook 2 logic) ---
try:
    df_original = pd.read_csv('original_data.csv')
    df_scores = pd.read_csv('score.csv')
except FileNotFoundError:
    print("Error: Make sure 'original_data.csv' and 'score.csv' are in the folder.")
    exit()

df_clean = df_original.copy()

# A. Convert GRE Scores
gre_q_map = pd.Series(df_scores.newQ.values, index=df_scores.old).to_dict()
gre_v_map = pd.Series(df_scores.newV.values, index=df_scores.old).to_dict()

def convert_score_and_validate(score, score_map):
    if score > 170: return score_map.get(score, np.nan)
    elif score >= 130: return score
    else: return np.nan

df_clean['greQ_new'] = df_clean['greQ'].apply(lambda x: convert_score_and_validate(x, gre_q_map))
df_clean['greV_new'] = df_clean['greV'].apply(lambda x: convert_score_and_validate(x, gre_v_map))

# B. Normalize CGPA
df_clean['cgpaScale'] = df_clean['cgpaScale'].replace(0, np.nan)
df_clean['cgpa_norm'] = (df_clean['cgpa'] / df_clean['cgpaScale']) * 4.0
df_clean.loc[df_clean['cgpa'] == 0, 'cgpa_norm'] = np.nan

# C. Impute missing values
numeric_features = ['toeflScore', 'greQ_new', 'greV_new', 'greA', 'cgpa_norm', 'researchExp', 'industryExp']
for col in ['toeflScore']:
    df_clean[col] = df_clean[col].replace(0, np.nan)

# --- NEW: Calculate and store medians before filling ---
imputation_medians = {}
for col in numeric_features:
    median_val = df_clean[col].median()
    imputation_medians[col] = median_val # Store it
    df_clean[col] = df_clean[col].fillna(median_val) # Then fill

print("Data cleaning and imputation complete. Medians stored.")


# --- 2. Build Artifacts for Recommender (Notebook 1) ---
print("Building artifacts for Recommender Model...")
df_admitted = df_clean[df_clean['admit'] == 1].copy()
df_admitted['cgpa_normalized'] = (df_admitted['cgpa_norm'] / 4.0) * 10.0 # Recommender used 10-pt scale

user_univ_matrix = df_admitted.pivot_table(index='userName', columns='univName', values='admit').fillna(0)
df_admitted_encoded = pd.get_dummies(df_admitted, columns=['major', 'program'], drop_first=True)
major_cols = [c for c in df_admitted_encoded.columns if c.startswith('major_')]
program_cols = [c for c in df_admitted_encoded.columns if c.startswith('program_')]
advanced_profile_features = ['greQ_new', 'greV_new', 'greA', 'cgpa_normalized', 'researchExp', 'industryExp', 'toeflScore'] + major_cols + program_cols
user_profiles_advanced = df_admitted_encoded.set_index('userName')[advanced_profile_features].copy()
user_profiles_advanced.replace([np.inf, -np.inf], np.nan, inplace=True)
training_means_rec = user_profiles_advanced.mean().to_dict()
for col, mean in training_means_rec.items():
    user_profiles_advanced[col] = user_profiles_advanced[col].fillna(mean)

scaler_advanced = StandardScaler()
scaled_user_profiles_advanced = scaler_advanced.fit_transform(user_profiles_advanced)
avg_profile_per_uni = df_admitted_encoded.groupby('univName')[advanced_profile_features].mean()
scaled_avg_profile_per_uni = scaler_advanced.transform(avg_profile_per_uni)
scaled_avg_profile_df = pd.DataFrame(scaled_avg_profile_per_uni, index=avg_profile_per_uni.index, columns=advanced_profile_features)
unique_majors_rec = df_admitted['major'].dropna().unique().tolist()
unique_programs_rec = df_admitted['program'].dropna().unique().tolist()

# --- 3. Build Artifacts for Predictor (Notebook 2) ---
print("Building artifacts for Admission Predictor Model...")

# Feature Engineering
top_colleges = df_clean['ugCollege'].value_counts().nlargest(50).index
df_clean['ugCollege_grouped'] = df_clean['ugCollege'].where(df_clean['ugCollege'].isin(top_colleges), 'Other')
df_clean['major'] = df_clean['major'].fillna('Unknown')


# --- NEW: Pre-calculate correlation matrix for the Overview page heatmap ---
print("Calculating correlation matrix for overview page...")
numeric_features_for_corr = [
    'greQ_new', 'greV_new', 'greA', 'cgpa_norm', 
    'researchExp', 'industryExp', 'toeflScore'
]
# Ensure we only use columns that exist in df_clean to avoid errors
existing_numeric_features = [f for f in numeric_features_for_corr if f in df_clean.columns]
correlation_matrix = df_clean[existing_numeric_features].corr()


features_new = ['greQ_new', 'greV_new', 'greA', 'cgpa_norm', 'researchExp', 'industryExp', 'toeflScore', 'univName', 'program', 'ugCollege_grouped', 'major']
target = 'admit'
model_df_new = df_clean[features_new + [target]].dropna()

X_new = pd.get_dummies(model_df_new[features_new], columns=['univName', 'program', 'ugCollege_grouped', 'major'], drop_first=True)
y_new = model_df_new[target]

# Final Model Training
best_params = {'subsample': 0.8, 'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.2, 'colsample_bytree': 0.7, 'objective': 'binary:logistic', 'use_label_encoder': False, 'eval_metric': 'logloss'}
final_predictor_model = xgb.XGBClassifier(**best_params)
final_predictor_model.fit(X_new, y_new)

# --- NEW: Get Feature Importance from the final model for the Overview page ---
print("Extracting feature importances for overview page...")
feature_importances = final_predictor_model.feature_importances_
feature_names = X_new.columns
# Create a DataFrame for easy sorting and plotting
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)



# Get data needed for UI and prediction function
predictor_training_columns = X_new.columns
all_universities_pred = model_df_new['univName'].unique().tolist()
unique_majors_pred = model_df_new['major'].dropna().unique().tolist()
unique_programs_pred = model_df_new['program'].dropna().unique().tolist()
unique_ug_colleges_pred = model_df_new['ugCollege_grouped'].dropna().unique().tolist()

print("Model training and artifact creation complete.")

# --- 4. Save All Artifacts to a Single File ---
all_artifacts = {
    # Recommender assets
    "user_univ_matrix": user_univ_matrix,
    "user_profiles_advanced": user_profiles_advanced,
    "scaled_user_profiles_advanced": scaled_user_profiles_advanced,
    "scaler_advanced": scaler_advanced,
    "scaled_avg_profile_df": scaled_avg_profile_df,
    "advanced_profile_features_rec": advanced_profile_features,
    "training_means_rec": training_means_rec,
    "unique_majors_rec": unique_majors_rec,
    "unique_programs_rec": unique_programs_rec,
    "df_admitted": df_admitted_encoded,

    # Predictor assets
    "predictor_model": final_predictor_model,
    "predictor_training_columns": predictor_training_columns,
    "all_universities_pred": all_universities_pred,
    "unique_majors_pred": unique_majors_pred,
    "unique_programs_pred": unique_programs_pred,
    "unique_ug_colleges_pred": unique_ug_colleges_pred,
    "feature_importance_df": feature_importance_df,
    
    # Shared assets
    "df_clean": df_clean, # For overview
    "correlation_matrix": correlation_matrix,
    "imputation_medians": imputation_medians
}

with open("app_artifacts.pkl", "wb") as f:
    pickle.dump(all_artifacts, f)

print("\nSetup complete! All artifacts saved to 'app_artifacts.pkl'.")
print("You can now run the Streamlit app.")