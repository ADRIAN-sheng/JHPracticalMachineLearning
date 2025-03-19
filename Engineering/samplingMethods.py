from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import pandas as pd

# Define the resampling method
resampling_method = "SMOTE+undersampling"  # Options: "SMOTE", "BorderlineSMOTE", "ADASYN", "undersampling", "SMOTE+undersampling"

# Prepare feature matrix and target variable
X = df.drop(columns=['loan_status']).copy()
y = df['loan_status'].copy()

# Check for any missing values before resampling
assert X.isnull().sum().sum() == 0, "Warning: Missing values detected in X before resampling."

# Choose the resampling method
if resampling_method == "SMOTE":
    resampler = SMOTE(sampling_strategy='auto', random_state=42)
elif resampling_method == "BorderlineSMOTE":
    resampler = BorderlineSMOTE(sampling_strategy='auto', random_state=42)
elif resampling_method == "ADASYN":
    resampler = ADASYN(sampling_strategy='auto', random_state=42)
elif resampling_method == "undersampling":
    resampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
elif resampling_method == "SMOTE+undersampling":
    # First, oversample the minority class using SMOTE
    smote = SMOTE(sampling_strategy=0.5, random_state=42)  # Adjust to achieve a 50-50 balance
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Then, undersample the majority class to maintain the 50-50 balance
    undersampler = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
else:
    raise ValueError("Invalid resampling method. Choose 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'undersampling', or 'SMOTE+undersampling'.")

# Apply the selected resampling technique (if not already applied in SMOTE+undersampling)
if resampling_method not in ["SMOTE+undersampling"]:
    X_resampled, y_resampled = resampler.fit_resample(X, y)

# Reindex the resampled data
X_resampled.reset_index(drop=True, inplace=True)
y_resampled.reset_index(drop=True, inplace=True)


# Checking the new class distribution
print(f"Class distribution after {resampling_method}:")
print(y_resampled.value_counts(normalize=True))
print('Shape of X_resampled is ',X_resampled.shape)

# Convert back to DataFrame
df_train_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                pd.DataFrame(y_resampled, columns=['loan_status'])], axis=1)

# Display first few rows of the resampled training set
df_train_resampled.head()



print(df_train_resampled.duplicated().sum())
# Drop duplicates
df_train_resampled.drop_duplicates(inplace=True)