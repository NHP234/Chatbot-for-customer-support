import torch
import numpy as np
import re
import os
import joblib
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from custom_models import CustomSVM, CustomMultinomialNB

# --- Model Loading ---

MODELS_DIR = "saved_models"

def load_all_models():
    """
    Loads all pre-trained models (vectorizers, classifiers) from the disk.
    Returns a dictionary containing the loaded models.
    """
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"'{MODELS_DIR}' directory not found. "
                              f"Please run 'python train.py' first to train and save the models.")
    
    print("Loading pre-trained models...")
    try:
        models = {
            "svm": joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl')),
            "tfidf_vectorizer": joblib.load(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')),
            "label_encoder": joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl')),
        }
        print("Pre-trained models loaded successfully.")
        return models
    except FileNotFoundError as e:
        print(f"Error loading models: {e}. Make sure all model files exist.")
        return None

# Load models once when the module is imported
loaded_models = load_all_models()
if loaded_models:
    svm_model = loaded_models['svm']
    tfidf_vectorizer = loaded_models['tfidf_vectorizer']
    le = loaded_models['label_encoder']
else:
    # Handle case where models failed to load
    svm_model, tfidf_vectorizer, le = None, None, None

# --- LLM Pipeline Initialization ---

def initialize_llm_pipeline():
    """
    Initializes and returns the Hugging Face text generation pipeline,
    dynamically selecting the model based on GPU availability.
    """
    print("Initializing LLM pipeline...")
    try:
        # Check for GPU availability
        if torch.cuda.is_available():
            print("CUDA is available. Initializing with GPU support (4-bit quantization)...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model_name = "unsloth/Phi-4-mini-reasoning-unsloth-bnb-4bit"
            model_kwargs = {
                "quantization_config": quantization_config,
                "device_map": "auto"
            }
        else:
            print("CUDA not found. Initializing with CPU support...")
            model_name = "unsloth/Phi-3-mini-4k-instruct" # Standard model for CPU
            model_kwargs = {
                "torch_dtype": "auto",
                "device_map": "auto"
            }

        llm_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        llm_pipe = pipeline(
            "text-generation",
            model=llm_model,
            tokenizer=tokenizer,
            max_new_tokens=2048, # Reduced for faster response times
            temperature=0.7,
            do_sample=True
        )
        print("LLM pipeline initialized successfully.")
        return llm_pipe
    except Exception as e:
        print(f"Error initializing LLM pipeline: {e}")
        return None

# Initialize LLM once when the module is imported
pipe = initialize_llm_pipeline()

# --- Response Generation ---

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
instruction_suffix = " Your response must be direct and helpful, without any of your own thought processes or meta-commentary like '<think>...</think>'."
for intent in intent_prompts:
    intent_prompts[intent] += instruction_suffix

def preprocess(text):
    return re.sub(r'[^\w\s]', '', text.lower())

def generate_response(query):
    if not all([svm_model, tfidf_vectorizer, le, pipe]):
        return "I'm sorry, I'm not fully initialized. Please check the server logs."

    # Intent Classification
    processed_query = preprocess(query)
    tfidf_vec = tfidf_vectorizer.transform([processed_query]).toarray()
    intent_idx = svm_model.predict(tfidf_vec)[0]
    intent = le.inverse_transform([intent_idx])[0]
    system_prompt = intent_prompts.get(intent, intent_prompts['default'])

    # LLM Response Generation
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    try:
        response = pipe(messages)
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
            generated_output_list = response[0].get('generated_text')
            if isinstance(generated_output_list, list) and generated_output_list:
                final_message = generated_output_list[-1]
                if isinstance(final_message, dict) and final_message.get('role') == 'assistant':
                    assistant_response_text = final_message.get('content', '')
                    parts = assistant_response_text.rsplit('</think>', 1)
                    cleaned_response = parts[1] if len(parts) > 1 else assistant_response_text
                    return cleaned_response.strip()
    except Exception as e:
        print(f"Error during LLM response generation: {e}")
        return "I'm sorry, an error occurred while I was thinking."
    
    return "I'm sorry, I couldn't generate a proper response."


if __name__ == "__main__":
    print("\n--- Example Responses ---")
    # Ensure all necessary components are ready before running examples
    if pipe is not None and \
       hasattr(tfidf_vectorizer, 'vocabulary_') and tfidf_vectorizer.vocabulary_ and \
       hasattr(svm_model, 'classes_') and svm_model.classes_ is not None and len(svm_model.classes_) > 0 and \
       le is not None:
        print("Query: I need to cancel my order immediately!")
        print("Response:", generate_response("I need to cancel my order immediately!"))

        print("\nQuery: How do I track my refund status?")
        print("Response:", generate_response("How do I track my refund status?"))
    else:
        print("Skipping example responses: LLM pipeline not ready, vectorizer not initialized, SVM not trained, or LabelEncoder not ready.")

print("\nScript finished.")