# Customer Support Chatbot

This is a chatbot project built with Python and Flask, utilizing Machine Learning models (SVM, Naive Bayes) for user intent classification and a Large Language Model (LLM - Phi-4-mini) for response generation.

## Prerequisites

Before you begin, ensure you have installed:

*   [Python](https://www.python.org/downloads/) (version 3.8 or higher recommended)
*   [pip](https://pip.pypa.io/en/stable/installation/) (usually comes with Python)
*   [Git](https://git-scm.com/downloads/)

## Setup Instructions

1.  **Clone the repository to your local machine:**
    ```bash
    git clone https://github.com/NHP234/Chatbot-for-customer-support.git
    cd Chatbot-for-customer-support
    ```

2.  **Create and activate a virtual environment:**
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note on PyTorch:**
    The `requirements.txt` file includes `torch`. The `pip install` command above will attempt to install a default version of PyTorch.
    *   If you intend to use a **GPU (NVIDIA)**, you might need to install a specific PyTorch build compatible with your CUDA driver. Please refer to the official [PyTorch website](https://pytorch.org/get-started/locally/) for the correct installation command. For example:
        ```bash
        # pip uninstall torch torchvision torchaudio # Uninstall existing if any
        # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # (Example for CUDA 11.8)
        ```
    *   If you are using **CPU only**, the default installed version is usually sufficient.

4.  **Data:**
    The project uses a dataset from Hugging Face (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`). The `main.py` script will automatically download this dataset on its first run (or use a cached version if available). Ensure you have an internet connection when running for the first time to download the dataset.

## Running the Application

After completing the setup steps:

1.  **Start the Flask server:**
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to the provided address (usually `http://127.0.0.1:5000/`).

    **Note:** The first time you start the application, it might take a while due to dataset downloading (if not cached), data preprocessing, training of intent classification models (SVM, Naive Bayes), and initialization of the LLM pipeline.

## Project Structure

```
Chatbot-for-customer-support/
├── static/                 # Contains static files (CSS, JavaScript)
│   ├── css/style.css
│   └── js/script.js
├── templates/              # Contains HTML templates (Flask)
│   └── index.html
├── venv/                   # Virtual environment directory (ignored by .gitignore)
├── .gitignore              # Files and directories ignored by Git
├── app.py                  # Main Flask application file (web backend)
├── main.py                 # Contains the core AI logic (data processing, models, response generation)
├── requirements.txt        # List of required Python libraries
└── README.md               # This instructional file
```

## (Optional) Retraining Models

Currently, the intent classification models (SVM, Naive Bayes) and the LLM pipeline are initialized and trained (if necessary) each time the `app.py` application starts (because `main.py` is imported and executed). To optimize startup time for a production environment, you might consider:
1.  Training the models once.
2.  Saving the trained models (e.g., using `joblib`).
3.  Modifying `main.py` to load the saved models instead of retraining them.
An example of how this could be implemented is by running `python main.py train` (after adjusting `main.py` to support this command). 