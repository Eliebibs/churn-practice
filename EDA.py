import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Read the dataset
df = pd.read_csv('churnpractice/Customer-Churn-Records.csv')

# 1. Remove unnecessary columns
columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
df = df.drop(columns=columns_to_drop)

# 2. CreditScore - robust scaling then min max scaling
robust_scaler = RobustScaler()
minmax_scaler = MinMaxScaler()
df['CreditScore'] = robust_scaler.fit_transform(df[['CreditScore']])
df['CreditScore'] = minmax_scaler.fit_transform(df[['CreditScore']])

# 3. Geography - one hot encoding then convert to 0/1
df = pd.get_dummies(df, columns=['Geography'])
# Convert boolean to int (True/False to 1/0)
geography_columns = [col for col in df.columns if 'Geography_' in col]
for col in geography_columns:
    df[col] = df[col].astype(int)

# 4. Gender - binary encoding (Female: 0, Male: 1)
df['Gender'] = (df['Gender'] == 'Male').astype(int)

# 5, 6, 8, 12, 14, 16 - Min/Max scaling
minmax_columns = ['Age', 'Tenure', 'NumOfProducts', 'EstimatedSalary', 
                 'Satisfaction Score', 'Point Earned']
df[minmax_columns] = minmax_scaler.fit_transform(df[minmax_columns])

# 7. Balance - log transform, then min max scale
df['Balance'] = np.log1p(df['Balance'])  # log1p handles zero values
df['Balance'] = minmax_scaler.fit_transform(df[['Balance']])

# 15. Card Type - ordinal encode then min/max scaling
card_type_mapping = {
    'SILVER': 1,
    'GOLD': 2,
    'DIAMOND': 3,
    'PLATINUM': 4
}
df['Card Type'] = df['Card Type'].map(card_type_mapping)
df['Card Type'] = minmax_scaler.fit_transform(df[['Card Type']])

# Columns that need no transformation: 'HasCrCard', 'IsActiveMember', 'Exited', 'Complain'

# Save the preprocessed dataset
df.to_csv('churnpractice/preprocessed_churn_data.csv', index=False)

# Print the first few rows and shape of the preprocessed dataset
print("\nPreprocessed Dataset Shape:", df.shape)
print("\nPreprocessed Dataset Head:")
print(df.head())
print("\nColumns in preprocessed dataset:")
print(df.columns.tolist())
