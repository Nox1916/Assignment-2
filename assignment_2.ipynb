import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

# Step 1: Download the dataset
url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
df = pd.read_csv(url)

# Step 2: Convert the dataset into a balanced class dataset
# Assuming class label is 'Class', where 1 represents fraud and 0 represents non-fraud
fraud_df = df[df['Class'] == 1]
non_fraud_df = df[df['Class'] == 0].sample(n=len(fraud_df))
balanced_df = pd.concat([fraud_df, non_fraud_df])

# Step 3: Create five samples
sample_size = min(len(fraud_df), len(non_fraud_df))
samples = [balanced_df.sample(sample_size) for _ in range(5)]

# Step 4: Define ML models and sampling techniques
models = [DecisionTreeClassifier(), RandomForestClassifier(), SVC(), LogisticRegression(), DecisionTreeClassifier()]
samplings = [RandomOverSampler(), SMOTE(), RandomUnderSampler(), NearMiss(), SMOTETomek()]

# Step 5: Apply sampling techniques on ML models and evaluate accuracy
results = []
for model, sampling in zip(models, samplings):
    accuracies = []
    for sample in samples:
        X = sample.drop('Class', axis=1)
        y = sample['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        pipeline = Pipeline([('sampling', sampling), ('classifier', model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    results.append(accuracies)

# Print results
print("Sampling Techniques vs Models:")
for i, model in enumerate(['M1', 'M2', 'M3', 'M4', 'M5']):
    print(f"\nModel {model}:")
    for j, sampling in enumerate(['Sampling1', 'Sampling2', 'Sampling3', 'Sampling4', 'Sampling5']):
        print(f"{sampling}: {results[j][i]*100:.2f}% accuracy")
