import pandas as pd
import numpy as np
import re
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datasets import load_dataset
from custom_models import CustomSVM, CustomMultinomialNB

# --- Evaluation Function ---

def evaluate(model, X_test, y_test, model_name, class_names):
    """Prints classification report and F1 score for a given model."""
    print(f"\n--- {model_name} Performance ---")
    y_pred = model.predict(X_test)
    
    # Ensure class_names are available for the report
    if class_names is None or len(class_names) == 0:
        class_names = [str(i) for i in np.unique(y_test)]

    try:
        # Generate and print the classification report
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        print(report)
        
        # Calculate and print the weighted F1 score
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"F1 Score (weighted): {f1:.4f}")
        
    except ValueError as e:
        print(f"Could not generate classification report for {model_name}: {e}")
        print("This might be due to a mismatch in labels or target_names.")

# --- Main Training Function ---

def preprocess(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def train_and_save_models():
    """
    This function loads data, trains all necessary models (vectorizers, classifiers),
    and saves them to disk in the 'saved_models' directory.
    """
    # Create directory to save models
    output_dir = "saved_models"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Models will be saved in '{output_dir}' directory.")

    # 1. Load dataset
    print("Loading dataset...")
    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(ds['train'][:10000]).fillna("")
    print(f"Dataset loaded with {len(df)} entries.")

    texts = df['instruction'].tolist()
    processed_texts = [preprocess(text) for text in texts]

    # 2. Label encoding
    print("Encoding labels...")
    le = LabelEncoder()
    y = le.fit_transform(df['intent'])
    
    # IMPORTANT: Save the label encoder right after fitting
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.pkl'))
    print("LabelEncoder saved.")

    # 3. Feature Extraction
    print("Initializing vectorizers...")
    tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=True)
    count_vectorizer = CountVectorizer(min_df=2, lowercase=True)

    # 4. Data Splitting
    print("Splitting data into training and testing sets...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        processed_texts, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Fit vectorizers on TRAINING data only and transform both sets
    print("Fitting vectorizers and transforming data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_texts).toarray()
    X_test_tfidf = tfidf_vectorizer.transform(X_test_texts).toarray()
    
    X_train_bow = count_vectorizer.fit_transform(X_train_texts).toarray()
    X_test_bow = count_vectorizer.transform(X_test_texts).toarray()

    # 6. Save the FITTED vectorizers
    joblib.dump(tfidf_vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
    print("TfidfVectorizer saved.")
    joblib.dump(count_vectorizer, os.path.join(output_dir, 'bow_vectorizer.pkl'))
    print("CountVectorizer saved.")

    # 7. Model Training (on training set)
    print("Training Custom SVM...")
    svm = CustomSVM(C=1.0, gamma='auto', max_iter=100)
    svm.fit(X_train_tfidf, y_train)
    
    print("Training Custom Naive Bayes...")
    nb = CustomMultinomialNB(alpha=1.0)
    nb.fit(X_train_bow, y_train)

    # 8. Evaluation (on testing set)
    print("\n--- Model Evaluation ---")
    class_names = le.classes_ if hasattr(le, 'classes_') else None
    evaluate(svm, X_test_tfidf, y_test, "Custom SVM", class_names)
    evaluate(nb, X_test_bow, y_test, "Custom Naive Bayes", class_names)

    # 9. Save the TRAINED models
    print("\nSaving trained models...")
    joblib.dump(svm, os.path.join(output_dir, 'svm_model.pkl'))
    print("Custom SVM model saved.")
    joblib.dump(nb, os.path.join(output_dir, 'nb_model.pkl'))
    print("Custom Naive Bayes model saved.")

    print("\n--- Training, evaluation, and saving complete! ---")
    print("You can now run the main application.")

if __name__ == '__main__':
    train_and_save_models() 