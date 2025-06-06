import torch
import numpy as np
import pandas as pd
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Custom SVM with RBF Kernel (One-vs-Rest)
class CustomSVM:
    def __init__(self, C=1.0, gamma='auto', max_iter=100): # Note: max_iter=100 is low for SVMs, may lead to underfitting.
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifiers = []
        self.classes_ = None
        self.support_vectors_ = None # Initialize attribute
        self.dual_coef_ = None # Initialize attribute
        self.intercept_ = None # Initialize attribute

    def _rbf_kernel(self, X1, X2):
        if self.gamma == 'auto':
            gamma_val = 1.0 / X1.shape[1] if X1.shape[1] > 0 else 1.0
        else:
            gamma_val = self.gamma
        pairwise_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1) - 2*np.dot(X1, X2.T)
        return np.exp(-gamma_val * pairwise_dists)

    def _fit_binary(self, X, y):
        n_samples, n_features = X.shape
        if n_features == 0: # Cannot train if no features
            # Set dummy values or handle as appropriate
            self.support_vectors_ = np.array([]).reshape(0,0)
            self.dual_coef_ = np.array([]).reshape(1,0)
            self.intercept_ = 0
            print("Warning: SVM binary fit called with no features.")
            return

        K = self._rbf_kernel(X, X)
        alpha = np.zeros(n_samples)

        # Simplified SMO implementation
        for _ in range(self.max_iter):
            for i in range(n_samples):
                Ei = np.dot(alpha * y, K[i]) - y[i] # K[i] is K[:,i] or K[i,:] due to K being symmetric
                if (y[i]*Ei < -0.001 and alpha[i] < self.C) or \
                   (y[i]*Ei > 0.001 and alpha[i] > 0):
                    j = np.random.randint(0, n_samples)
                    while j == i: # Ensure j is different from i
                        j = np.random.randint(0, n_samples)

                    Ej = np.dot(alpha * y, K[j]) - y[j]

                    old_alpha_i, old_alpha_j = alpha[i], alpha[j]

                    if y[i] != y[j]:
                        L = max(0, alpha[j] - alpha[i])
                        H = min(self.C, self.C + alpha[j] - alpha[i])
                    else:
                        L = max(0, alpha[i] + alpha[j] - self.C)
                        H = min(self.C, alpha[i] + alpha[j])

                    if L == H:
                        continue

                    eta = 2*K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue

                    alpha[j] = alpha[j] - y[j]*(Ei - Ej)/eta
                    alpha[j] = max(L, min(H, alpha[j]))

                    if abs(alpha[j] - old_alpha_j) < 1e-5: # Threshold for change
                        continue

                    alpha[i] = alpha[i] + y[i]*y[j]*(old_alpha_j - alpha[j])

        # Store support vectors
        sv_idx = alpha > 1e-5
        if not np.any(sv_idx): # No support vectors found
             # Fallback: could use all points or handle differently.
             # For now, indicates a problem or very simple dataset for this class
            print(f"Warning: No support vectors found for a binary class. Model might not be effective.")
            # Set dummy values to prevent errors, though this classifier will be poor
            self.support_vectors_ = X # Use all X as a fallback, or an empty array
            self.dual_coef_ = (alpha * y).reshape(1, -1)
            # Intercept calculation might be problematic if no true SVs
            # A simple intercept might be the mean of y, or 0 if balanced.
            self.intercept_ = np.mean(y - np.dot(self.dual_coef_, K)) if np.any(alpha) else 0
            return


        self.support_vectors_ = X[sv_idx]
        self.dual_coef_ = (alpha[sv_idx] * y[sv_idx]).reshape(1, -1)

        # Intercept calculation
        # Decision boundary for SVs should be y_s (f(x_s)) = 1 or -1
        # f(x_s) = sum(alpha_i y_i K(x_i, x_s)) + b
        # So b = y_s - sum(alpha_i y_i K(x_i, x_s))
        # Average b over non-bound support vectors (0 < alpha_s < C) for robustness
        non_bound_sv_idx = sv_idx & (alpha > 1e-5) & (alpha < self.C - 1e-5)
        if np.any(non_bound_sv_idx):
            idx_to_use = non_bound_sv_idx
        elif np.any(sv_idx): # If no non-bound SVs, use all SVs (less robust)
            idx_to_use = sv_idx
        else: # Should not happen if we returned earlier with a warning
            self.intercept_ = 0
            return

        # Kernel matrix between all X and the support vectors X[idx_to_use]
        kernel_for_intercept = self._rbf_kernel(X[idx_to_use], self.support_vectors_)
        # Predictions for these SVs without intercept
        preds_without_intercept = np.dot(kernel_for_intercept, self.dual_coef_.T).flatten()
        self.intercept_ = np.mean(y[idx_to_use] - preds_without_intercept)


    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = [] # Reset classifiers
        if X.shape[0] == 0 or X.shape[1] == 0:
            print("Warning: SVM fit called with empty data or no features. Model will not be trained.")
            self.classes_ = np.array([]) # Ensure classes_ is empty if not trained
            return

        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            # Each classifier is an instance of this class, but will store its own SVs, dual_coef, intercept
            classifier_instance = CustomSVM(C=self.C, gamma=self.gamma, max_iter=self.max_iter) # Create a new base instance for binary task
            classifier_instance._fit_binary(X, y_binary) # This sets SVs etc. on classifier_instance
            self.classifiers.append(classifier_instance) # Append the trained binary classifier

    def decision_function(self, X):
        if not self.classifiers or X.shape[1] == 0 : # No classifiers trained or no features in X
            return np.zeros((X.shape[0], len(self.classes_ if self.classes_ is not None else [])))

        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls_trainer in enumerate(self.classifiers):
            if cls_trainer.support_vectors_ is None or cls_trainer.support_vectors_.shape[0] == 0 or cls_trainer.support_vectors_.shape[1] != X.shape[1]:
                # This binary classifier was not trained properly or feature mismatch
                scores[:, i] = 0 # Default score
                continue
            K = self._rbf_kernel(X, cls_trainer.support_vectors_)
            scores[:, i] = np.dot(K, cls_trainer.dual_coef_.T).flatten() + cls_trainer.intercept_
        return scores

    def predict(self, X):
        if not self.classifiers or (self.classes_ is not None and len(self.classes_) == 0):
            # Model not trained or no classes, return empty or raise error
            # For safety, predict first class or a default if X has rows, else empty.
            # This depends on desired behavior for untrained model.
            print("Warning: SVM predict called on an untrained model or model with no classes.")
            return np.array([0] * X.shape[0]) # Predict a default class (e.g., first class if known)

        decision_scores = self.decision_function(X)
        if decision_scores.shape[1] == 0: # No scores generated
            return np.array([0] * X.shape[0])
        return self.classes_[np.argmax(decision_scores, axis=1)]

# Custom Multinomial Naive Bayes
class CustomMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None
        self.class_log_prior_ = None
        self.feature_log_prob_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if n_samples == 0 or n_features == 0:
            print("Warning: Naive Bayes fit called with empty data or no features. Model will not be trained.")
            return

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Calculate class log priors
        class_counts = np.bincount(y, minlength=n_classes) # Ensure all classes are counted even if some are not in y for this specific call
        self.class_log_prior_ = np.log((class_counts + 1e-8) / (n_samples + 1e-8 * n_classes)) # Add epsilon for stability

        # Calculate feature log probabilities
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        # Map class values to indices 0 to n_classes-1 for bincount and direct indexing
        class_map = {cls_val: i for i, cls_val in enumerate(self.classes_)}

        for cls_val in self.classes_:
            cls_idx = class_map[cls_val]
            X_cls = X[y == cls_val]
            if X_cls.shape[0] == 0: # No samples for this class
                # Assign a very small probability (or uniform) to avoid issues.
                # This is a form of smoothing if a class is present in `classes_` but has no training samples.
                self.feature_log_prob_[cls_idx] = np.log((self.alpha) / (n_features * self.alpha)) # Smoothed uniform
                continue

            total_count_per_feature = X_cls.sum(axis=0) + self.alpha
            self.feature_log_prob_[cls_idx] = np.log(total_count_per_feature / (total_count_per_feature.sum())) # Removed + n_features * self.alpha from denom as it's already in total_count_per_feature

    def predict(self, X):
        if self.feature_log_prob_ is None or self.class_log_prior_ is None or self.classes_ is None:
            print("Warning: Naive Bayes predict called on an untrained model.")
            return np.array([0] * X.shape[0]) # Predict a default class

        if X.shape[1] != self.feature_log_prob_.shape[1]:
            print(f"Error: Feature count mismatch in Naive Bayes predict. Expected {self.feature_log_prob_.shape[1]}, got {X.shape[1]}.")
            # Fallback: predict default class for all samples
            return np.array([self.classes_[0] if len(self.classes_) > 0 else 0] * X.shape[0])


        jll = X @ self.feature_log_prob_.T + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]

# Preprocessing for text
def preprocess(text):
    return re.sub(r'[^\w\s]', '', text.lower())

# Load dataset
print("Loading dataset...")
ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(ds['train'][:10000]).fillna("") # Using a subset for potentially faster processing, adjust as needed
print(f"Dataset loaded with {len(df)} entries.")

texts = df['instruction'].tolist()
processed_texts = [preprocess(text) for text in texts]

# Feature extraction using TfidfVectorizer
print("Extracting features using TfidfVectorizer...")
tfidf_vectorizer = TfidfVectorizer(min_df=2, lowercase=True)
X_tfidf = tfidf_vectorizer.fit_transform(processed_texts).toarray()
print(f"TF-IDF features extracted with shape: {X_tfidf.shape}")

# Also create BOW features for Naive Bayes
print("Extracting Bag-of-Words features...")
count_vectorizer = CountVectorizer(min_df=2, lowercase=True)
X_bow = count_vectorizer.fit_transform(processed_texts).toarray()
print(f"BOW features extracted with shape: {X_bow.shape}")

# Label encoding
le = LabelEncoder()
y = le.fit_transform(df['intent'])

# Stratified split
print("Splitting data...")
if X_tfidf.shape[0] > 0 and X_bow.shape[0] > 0 and len(y) > 0:
    X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train_bow, X_test_bow, _, _ = train_test_split(
        X_bow, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Data split successfully.")
else:
    print("Error: Not enough data to perform train/test split. Models will not be trained.")
    # Create empty arrays to prevent downstream errors
    X_train_tfidf, X_test_tfidf, y_train, y_test = np.array([]), np.array([]), np.array([]), np.array([])
    X_train_bow, X_test_bow = np.array([]), np.array([])

# Model Training
print("\n--- Model Training ---")
svm = CustomSVM(C=1.0, gamma='auto', max_iter=100)
if X_train_tfidf.shape[0] > 0 and X_train_tfidf.shape[1] > 0 and len(y_train) > 0:
    print("Training Custom SVM...")
    svm.fit(X_train_tfidf, y_train)
    print("Custom SVM training complete.")
else:
    print("Skipping Custom SVM training: Not enough training data or features.")

nb = CustomMultinomialNB(alpha=1.0)
if X_train_bow.shape[0] > 0 and X_train_bow.shape[1] > 0 and len(y_train) > 0:
    print("Training Custom Naive Bayes...")
    nb.fit(X_train_bow, y_train)
    print("Custom Naive Bayes training complete.")
else:
    print("Skipping Custom Naive Bayes training: Not enough training data or features.")

# Evaluation
def evaluate(model, X, y_true, model_name, class_names):
    print(f"\n--- {model_name} Performance ---")
    if not hasattr(model, 'classes_') or model.classes_ is None or len(model.classes_) == 0:
        print(f"Model not trained or has no classes. Skipping evaluation for {model_name}.")
        return

    if X.shape[0] == 0:
        print(f"Test data X is empty. Skipping evaluation for {model_name}.")
        return

    if X.shape[1] == 0 and isinstance(model, CustomSVM) and (model.support_vectors_ is not None and model.support_vectors_.shape[1] != 0):
        print(f"Test data X has no features, but model expects features. Skipping evaluation for {model_name}.")
        return

    if isinstance(model, CustomMultinomialNB) and model.feature_log_prob_ is not None and X.shape[1] != model.feature_log_prob_.shape[1]:
        print(f"Feature mismatch for Naive Bayes. Model expects {model.feature_log_prob_.shape[1]} features, X has {X.shape[1]}. Skipping.")
        return


    y_pred = model.predict(X)

    if len(y_true) == 0 or len(y_pred) == 0:
        print(f"True labels or predictions are empty. Skipping detailed report for {model_name}.")
        return

    # Ensure class_names match the unique labels present in y_true and y_pred for the report
    # This is important if some classes were not in y_train but appear in y_test, or vice-versa
    active_classes_indices = np.unique(np.concatenate((y_true, y_pred)))

    # Ensure target names are available and cover all active classes
    if class_names is None or len(class_names) == 0:
        # Fallback to stringified numerical labels if le.classes_ wasn't available or empty
        report_class_names = [str(i) for i in active_classes_indices]
        # It's better to ensure le.classes_ is correctly passed and used
    else:
        # Filter class_names to only those present in y_true/y_pred if needed,
        # or ensure that class_names covers all possible values from le.
        # For classification_report, target_names should correspond to sorted unique labels in y_true, y_pred.
        # Or it uses labels parameter. Best to pass what le.classes_ provides.
        report_class_names = class_names # Assuming le.classes_ is comprehensive

    try:
        print(classification_report(y_true, y_pred, target_names=report_class_names, labels=np.arange(len(report_class_names)), zero_division=0))
        print(f"F1 Score (weighted): {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    except ValueError as e:
        print(f"Could not generate classification report for {model_name}: {e}")
        print("This might be due to a mismatch in labels or target_names.")
        print(f"Unique y_true: {np.unique(y_true)}, Unique y_pred: {np.unique(y_pred)}")
        print(f"Report class names provided: {report_class_names}")


le_classes = le.classes_ if hasattr(le, 'classes_') else None
evaluate(svm, X_test_tfidf, y_test, "Custom SVM", le_classes)
evaluate(nb, X_test_bow, y_test, "Custom Naive Bayes", le_classes)


# Response Generation Pipeline
print("\n--- Setting up Response Generation Pipeline ---")
intent_prompts = {
    'cancel_order': "You are a customer support assistant. Apologize sincerely and guide through cancellation process step-by-step.",
    'change_order': "You are a customer support assistant. Explain order modification options clearly and ask for order ID.",
    'change_shipping_address': "You are a customer support assistant. Provide address change form link and verify order details.",
    'check_cancellation_fee': "You are a customer support assistant. Explain fee structure based on order timeline clearly.",
    'check_invoice': "You are a customer support assistant. Provide invoice access instructions and offer resend option.",
    'check_payment_methods': "You are a customer support assistant. List available payment methods with security assurances.",
    'check_refund_policy': "You are a customer support assistant. Explain refund policy timeline and conditions clearly.",
    'complaint': "You are a customer support assistant. Apologize empathetically and escalate to supervisor if needed.",
    'contact_customer_service': "You are a customer support assistant. Provide contact options with expected response times.",
    'contact_human_agent': "You are a customer support assistant. Offer callback option while explaining wait times.",
    'create_account': "You are a customer support assistant. Guide through account creation with security tips.",
    'delete_account': "You are a customer support assistant. Explain account deletion process and alternatives.",
    'delivery_options': "You are a customer support assistant. List available delivery methods with pricing.",
    'delivery_period': "You are a customer support assistant. Provide delivery estimates with tracking instructions.",
    'edit_account': "You are a customer support assistant. Explain account settings modification process.",
    'get_invoice': "You are a customer support assistant. Provide secure invoice download link and verify identity.",
    'get_refund': "You are a customer support assistant. Guide through refund process with timeline expectations.",
    'newsletter_subscription': "You are a customer support assistant. Explain subscription preferences management.",
    'payment_issue': "You are a customer support assistant. Troubleshoot payment methods securely.",
    'place_order': "You are a customer support assistant. Confirm order details and provide next steps.",
    'recover_password': "You are a customer support assistant. Guide through secure password reset process.",
    'registration_problems': "You are a customer support assistant. Troubleshoot registration errors step-by-step.",
    'review': "You are a customer support assistant. Thank for feedback and offer escalation if needed.",
    'set_up_shipping_address': "You are a customer support assistant. Guide through address book management.",
    'switch_account': "You are a customer support assistant. Explain account switching process securely.",
    'track_order': "You are a customer support assistant. Provide tracking portal link and status explanation.",
    'track_refund': "You are a customer support assistant. Provide refund tracking system access and timeline.",
    'default': "You are a helpful customer support assistant. Respond professionally to the inquiry."
}

# Add an instruction to all prompts to prevent the model from showing its thought process.
# This is the primary fix for inconsistent output containing <think> tags.
instruction_suffix = " Your response must be direct and helpful, without any of your own thought processes, meta-commentary, or self-correction like '<think>...</think>'."
for intent in intent_prompts:
    intent_prompts[intent] += instruction_suffix


pipe = None
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model_name = "unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit" # Changed from Phi-3 to Phi-4 as in original
    llm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=tokenizer,
        max_new_tokens=8192, # Controls length of generated text
        temperature=0.7,
        do_sample=True
    )
    print("Text generation pipeline initialized successfully.")
except Exception as e:
    print(f"Error initializing text generation pipeline: {e}")
    print("Response generation will be affected.")


def generate_response(query):
    if not hasattr(tfidf_vectorizer, 'vocabulary_') or not tfidf_vectorizer.vocabulary_:
        print("Error: TF-IDF vectorizer not properly initialized. Cannot process query.")
        system_prompt = intent_prompts['default']  # Fallback to default prompt
    elif not hasattr(svm, 'classes_') or svm.classes_ is None or len(svm.classes_) == 0:
        print("Warning: SVM model not trained or ready. Falling back to default intent.")
        system_prompt = intent_prompts['default']
    else:
        # Preprocess and transform the input query
        processed_query = preprocess(query)
        try:
            tfidf_vec = tfidf_vectorizer.transform([processed_query]).toarray()
            intent_idx = svm.predict(tfidf_vec)[0]
            intent = le.inverse_transform([intent_idx])[0]
            system_prompt = intent_prompts.get(intent, intent_prompts['default'])
        except Exception as e:
            print(f"Error during query vectorization or prediction: {e}")
            system_prompt = intent_prompts['default']

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]

    if pipe is None:
        print("Error: Text generation pipeline is not available.")
        return "I'm sorry, I cannot generate a response at this time (pipeline not ready)."

    try:
        response = pipe(messages)  # max_new_tokens is set in pipeline
        print(f"DEBUG main.py - Full response from LLM pipeline: {response}")
        print(f"DEBUG main.py - Type of full response: {type(response)}")
        if isinstance(response, list) and len(response) > 0:
            print(f"DEBUG main.py - First element of response: {response[0]}")
            print(f"DEBUG main.py - Type of first element: {type(response[0])}")
            if isinstance(response[0], dict):
                print(f"DEBUG main.py - Keys in first element: {response[0].keys()}")
                generated_output_list = response[0].get('generated_text')
                print(f"DEBUG main.py - Raw generated_text output (should be a list of dicts): {generated_output_list}")
                print(f"DEBUG main.py - Type of raw generated_text output: {type(generated_output_list)}")
                
                if isinstance(generated_output_list, list) and generated_output_list:
                    # Tìm phản hồi của assistant, thường là phần tử cuối cùng hoặc phần tử có role 'assistant'
                    assistant_response_text = ""
                    # Cách 1: Giả sử assistant response luôn là content của dict cuối cùng trong list
                    # last_turn = generated_output_list[-1]
                    # if isinstance(last_turn, dict) and last_turn.get('role') == 'assistant':
                    #    assistant_response_text = last_turn.get('content', "Error: 'content' key missing in assistant turn.")
                    
                    # Cách 2: Lặp qua list để tìm dict có role 'assistant' và lấy content gần nhất với user query.
                    # Điều này linh hoạt hơn nếu output có thể chứa nhiều lượt assistant hoặc cấu trúc phức tạp hơn.
                    # Tuy nhiên, với cấu trúc hiện tại, LLM trả về toàn bộ cuộc hội thoại bao gồm cả system prompt và user query.
                    # Chúng ta chỉ muốn phần mới được tạo bởi assistant.
                    # Thông thường, pipeline text-generation trả về toàn bộ input messages + generated text.
                    # Ta cần phần text sau cùng mà không phải là input.
                    
                    # Cách tiếp cận đơn giản nhất dựa trên output quan sát được:
                    # output từ Phi-4-mini-reasoning-unsloth-bnb-4bit có vẻ trả về toàn bộ chuỗi messages.
                    # Phần cuối cùng của messages list NÊN là câu trả lời của assistant.
                    final_message_in_sequence = generated_output_list[-1]
                    if isinstance(final_message_in_sequence, dict) and final_message_in_sequence.get('role') == 'assistant':
                        assistant_response_text = final_message_in_sequence.get('content')
                        if assistant_response_text:
                            # The model consistently places the final, clean response after the last </think> tag.
                            # We will split the string at the last occurrence of '</think>' and take the part that follows.
                            parts = assistant_response_text.rsplit('</think>', 1)

                            # If the split was successful, the response is the second part. Otherwise, use the whole text.
                            if len(parts) > 1:
                                cleaned_response_text = parts[1]
                            else:
                                # Fallback in case the model behaves and doesn't produce <think> tags.
                                cleaned_response_text = assistant_response_text

                            # Remove any leading characters like '---', whitespace, or other special tokens.
                            cleaned_response_text = re.sub(r'^[\s\W]*---', '', cleaned_response_text.strip())
                            
                            # Remove any remaining unwanted artifacts and trailing whitespace.
                            cleaned_response_text = re.sub(r'<\|.*?\|>', '', cleaned_response_text) # Remove special tokens like <|...|>
                            cleaned_response_text = re.sub(r'\\boxed{\\text{.*?}}', '', cleaned_response_text).strip()
                            
                            print(f"DEBUG main.py - Cleaned Markdown response: {cleaned_response_text}")
                            print(f"DEBUG main.py - Type of Cleaned Markdown response: {type(cleaned_response_text)}")
                            return str(cleaned_response_text)
                        else:
                            print("DEBUG main.py - Assistant content is empty or None.")
                            return "Error: Assistant content not found or empty."
                    else:
                        print("DEBUG main.py - Last message in sequence is not from assistant or not a dict.")
                        return "Error: Could not isolate assistant response from LLM output."
                else:
                    print("DEBUG main.py - 'generated_text' is not a list or is empty.")
                    return "Error: LLM response format unexpected ('generated_text' not a list or empty)."
            else:
                print("DEBUG main.py - First element is not a dictionary.")
                return "Error: LLM response format unexpected (first element not a dict)."
        else:
            print("DEBUG main.py - LLM response is not a list or is empty.")
            return "Error: LLM response format unexpected (not a list or empty)."
    except Exception as e:
        print(f"Error during text generation: {e}")
        return "I'm sorry, an error occurred while generating the response."


if __name__ == "__main__":
print("\n--- Example Responses ---")
    # Ensure all necessary components are ready before running examples
    if pipe is not None and \
       hasattr(tfidf_vectorizer, 'vocabulary_') and tfidf_vectorizer.vocabulary_ and \
       hasattr(svm, 'classes_') and svm.classes_ is not None and len(svm.classes_) > 0 and \
       le is not None:
    print("Query: I need to cancel my order immediately!")
    print("Response:", generate_response("I need to cancel my order immediately!"))

    print("\nQuery: How do I track my refund status?")
    print("Response:", generate_response("How do I track my refund status?"))
else:
        print("Skipping example responses: LLM pipeline not ready, vectorizer not initialized, SVM not trained, or LabelEncoder not ready.")

print("\nScript finished.")