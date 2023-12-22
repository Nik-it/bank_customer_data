# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Import Support Vector Machine classifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
from sklearn.metrics import accuracy_score


# Load the dataset
data = pd.read_csv('/home/nikit/djangochat/bank_customer_data.csv')

# Preprocess the data and split into features and target
X = data.drop('Services_Taken', axis=1)
y = data['Services_Taken']

# Add missing columns with zeros
additional_columns = ['column1', 'column2', 'column3', 'column4', 'column5']  # Add your column names
X[additional_columns] = 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Define the preprocessing steps
numeric_features = ['User_ID', 'Recent_Activities', 'Purchase_History', 'Money_on_Account', 'Age', 'Monthly_Income', 'Credit_Score', 'Number_of_Dependents']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Account_Type', 'Education_Level', 'Gender']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model (replace RandomForestClassifier with SVC)
model = SVC(probability=True)  # Enable probability estimates

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Now, you can make predictions
predictions = pipeline.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')


# Save the trained pipeline
joblib.dump(pipeline, '/home/nikit/djangochat/your_trained_model.pkl')
