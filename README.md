# Clinical Text Classification with Explainable AI (XAI)

## 1. Executive Summary

This project implements a production-ready prototype for classifying clinical notes using a machine learning pipeline, with a core focus on providing robust, model-agnostic explainability. In high-stakes environments like clinical trials and diagnostics, model transparency is a prerequisite for adoption. This system directly addresses the "black box" problem by integrating SHAP (SHapley Additive exPlanations) to deliver clear, word-level attributions for every prediction, fostering trust and enabling critical validation by domain experts.

The final deliverable is an interactive Dash dashboard designed for clinical researchers and data scientists. It provides a user interface for submitting clinical text, receiving a classification, and visualizing the underlying drivers of the model's decision.

---

## 2. System Architecture & Technical Pipeline

The system is architected as a modular, end-to-end pipeline, ensuring reproducibility and maintainability. Each component is a discrete Python script, designed to be executed in sequence or orchestrated via the `main.py` script.

![Interactive Dashboard](https://github.com/sephcodes66/Clinical-Notes-XAI-/blob/main/ss/interactive_dashboard1.png)

### 2.1. Data Processing & Modeling Workflow

1.  **Data Preparation (`data_preparation.py`):** Ingests raw data (`mtsamples.csv`) and applies standardized preprocessing steps to clean and structure the text for downstream feature extraction.
2.  **Feature Extraction (`feature_extraction.py`):** Leverages a pre-trained transformer model, `emilyalsentzer/Bio_ClinicalBERT`, to generate high-fidelity, 768-dimensional embeddings. This specific model is chosen for its strong performance on biomedical and clinical text, ensuring a nuanced semantic representation.
3.  **Classifier Training (`train_classifier.py`):** A Logistic Regression model is trained on the embeddings. This classifier was selected for its computational efficiency, inherent interpretability at the feature level, and robust performance, making it an excellent baseline for this task. The modular design permits swapping this component with more complex models (e.g., XGBoost, Neural Networks) as required.
4.  **Explainability Model Generation (`explainability.py`):** Constructs and serializes a SHAP explainer object. This object encapsulates the entire pipeline (embedding and classification) to ensure that explanations account for the complete transformation from raw text to prediction.

### 2.2. Core Technologies

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Language Model** | Hugging Face Transformers (Bio_ClinicalBERT) | State-of-the-art semantic understanding of clinical terminology. |
| **Classification** | Scikit-learn (Logistic Regression) | Excellent balance of performance and speed; robust and well-understood. |
| **Explainability** | SHAP (SHapley Additive exPlanations) | Provides mathematically-grounded, model-agnostic, and locally accurate feature attributions. |
| **Dashboard** | Plotly / Dash | Industry-standard for building interactive, data-driven analytical applications in Python. |
| **Serialization** | Joblib | Efficient persistence of Python objects (models, explainers), optimized for NumPy-like structures. |

---

## 3. Operational Guide

### 3.1. Prerequisites

- Python 3.8+
- `venv` for environment management

### 3.2. Setup & Installation

1.  **Clone the repository and navigate to the project root.**

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv_healthai
    source venv_healthai/bin/activate
    ```
    *Note: On Windows, use `venv_healthai\Scripts\activate`.*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the dataset** `mtsamples.csv` into the `data/raw/` directory.

### 3.3. Pipeline Execution

Two execution methods are provided:

**A) Automated Orchestration (Recommended)**

The `main.py` script executes the core data processing and model training pipeline in the correct sequence.

```bash
python src/main.py
```

**B) Manual Step-wise Execution (For Debugging & Development)**

For granular control or debugging, run the scripts individually from the project root directory in this specific order:

```bash
python src/data_preparation.py
python src/feature_extraction.py
python src/train_classifier.py
python src/explainability.py
```

### 3.4. Launching the Application

Once the pipeline has been executed and the models are built, launch the interactive dashboard:

```bash
python src/app.py
```
The application will be available at `http://127.0.0.1:8050/`.

---

## 4. The Explainability Framework: Local vs. Global Insights

This system provides two distinct but complementary forms of explainability, catering to different analytical needs.

-   **Local Explanations (`app.py`):** The interactive dashboard delivers real-time, instance-specific explanations. It answers the question: **"Why was *this specific* clinical note classified this way?"** This is essential for case-by-case analysis, error auditing, and building trust with end-users.

-   **Global Explanations (`visualization.py`):** This script provides a high-level, aggregate view of the model's behavior. It answers the question: **"Across the entire dataset, what features does the model deem most important for each class?"** It achieves this by averaging SHAP values over a sample of the data, offering a powerful tool for model validation and ensuring it has learned clinically relevant patterns rather than spurious correlations.

---

## 5. Future Roadmap & Scalability

This prototype serves as a robust foundation. The following areas are targeted for future development:

1.  **Domain-Specific Fine-Tuning:** Fine-tune the ClinicalBERT model on the target dataset to potentially enhance classification accuracy and adapt its representations to the specific clinical sub-domain.
2.  **Model Performance Benchmarking:** Systematically evaluate alternative classifiers (e.g., XGBoost, LightGBM) to benchmark performance, latency, and explainability trade-offs.
3.  **Production Deployment:** Containerize the application using Docker and deploy it to a cloud service (e.g., AWS, Azure, GCP). This includes exploring optimized model serving frameworks (like TorchServe or ONNX Runtime) to improve inference speed.
4.  **Model Distillation:** For applications requiring very low latency, explore model distillation techniques to create a smaller, faster model that mimics the behavior of the larger ClinicalBERT-based pipeline while retaining a high degree of accuracy.

---

## 6. Licenses
This project is licensed under the MIT License.

The following libraries are used in this project, and their licenses are listed below:

| Library | License | Link |
| --- | --- | --- |
| pandas | BSD 3-Clause | https://github.com/pandas-dev/pandas/blob/main/LICENSE |
| numpy | BSD 3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| scikit-learn | BSD 3-Clause | https://github.com/scikit-learn/scikit-learn/blob/main/COPYING |
| joblib | BSD 3-Clause | https://github.com/joblib/joblib/blob/main/LICENSE.txt |
| torch | BSD | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| transformers | Apache 2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| huggingface-hub | Apache 2.0 | https://github.com/huggingface/huggingface_hub/blob/main/LICENSE |
| shap | MIT | https://github.com/slundberg/shap/blob/master/LICENSE |
| plotly | MIT | https://github.com/plotly/plotly.py/blob/master/LICENSE.txt |
| dash | MIT | https://github.com/plotly/dash/blob/dev/LICENSE |
| dash-bootstrap-components | Apache 2.0 | https://github.com/facultyai/dash-bootstrap-components/blob/main/LICENSE |
| matplotlib | PSF-based | https://github.com/matplotlib/matplotlib/blob/main/LICENSE/LICENSE |
