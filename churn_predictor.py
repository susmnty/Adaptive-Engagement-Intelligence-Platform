import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import joblib

def load_data():
    df = pd.read_csv('user_segments_with_anomalies.csv')
    print("[✓] Loaded user_segments_with_anomalies.csv")

    features = ['total_sessions', 'avg_session_length', 'add_to_cart_rate', 
                'purchase_rate', 'avg_spend', 'unique_categories']

    # Check for missing columns
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing features in dataset: {missing}")

    # Define churn based on median session length
    churn_threshold = df['avg_session_length'].median()
    df['churn'] = np.where(df['avg_session_length'] < churn_threshold, 1, 0)

    return df, features

def compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVM (RBF Kernel)": SVC(probability=True),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Gaussian Naive Bayes": GaussianNB(),
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
        "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis()
    }

    results = []

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            results.append((name, acc, report, model))
            print(f"\n[✓] {name} Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"\n[✗] {name} failed: {e}")
    
    return results

def main():
    df, features = load_data()

    X = df[features]
    y = df['churn']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Compare models
    results = compare_models(X_train, X_test, y_train, y_test)

    # Print best model
    best_model_info = max(results, key=lambda x: x[1])
    best_name, best_acc, best_report, best_model = best_model_info
    print(f"\n🔝 Best Model: {best_name} with Accuracy: {best_acc:.4f}")
    print("\nFull Classification Report:")
    print(pd.DataFrame(best_report).transpose())

    # Save the best model to disk
    model_filename = f"{best_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(best_model, model_filename)
    print(f"\n[✓] Saved best model as '{model_filename}'")

    # Save updated data with churn column
    df.to_csv('user_segments_with_churn_and_anomalies.csv', index=False)
    print("[📁] Saved updated data as 'user_segments_with_churn_and_anomalies.csv'")

if __name__ == '__main__':
    main()