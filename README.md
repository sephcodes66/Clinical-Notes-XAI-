# Clinical Text Classification with Explainable AI (XAI)

I built this project to explore how to make machine learning models in healthcare more transparent. I've always been fascinated by the potential of AI in medicine, but I'm also wary of the "black box" problem. How can we trust a model's prediction if we don't know why it's making it? This project is my attempt to answer that question.

This system uses a machine learning pipeline to classify clinical notes, and the cool part is that it uses SHAP (SHapley Additive exPlanations) to show exactly which words in the text led to the model's decision. This way, doctors and researchers can see for themselves if the model is making sense.

The end result is an interactive dashboard where you can paste in a clinical note, get a classification, and see a visual explanation of the result.

---

## So, How Does It Work?

The whole thing is set up as a series of Python scripts that run one after the other. You can run them all at once with `main.py` or step-by-step if you want to see what's happening under the hood.

![Interactive Dashboard](https://github.com/sephcodes66/Clinical-Notes-XAI-/blob/main/ss/interactive_dashboard1.png)

### The Pipeline

1.  **Data Prep (`data_preparation.py`):** First, we take the raw data (`mtsamples.csv`) and clean it up. This gets rid of any weird formatting and gets the text ready for the next step.
2.  **Feature Extraction (`feature_extraction.py`):** This is where the magic happens. I'm using a pre-trained model called `emilyalsentzer/Bio_ClinicalBERT` to turn the text into numerical embeddings. I chose this one because it's specifically trained on clinical text, so it understands the jargon.
3.  **Training the Classifier (`train_classifier.py`):** I went with a simple Logistic Regression model to classify the embeddings. It's fast, it's easy to understand, and it does a surprisingly good job. I thought about using something more complex like XGBoost, but I decided to keep it simple for this first version. You can always swap it out for a different model if you want.
4.  **Creating the Explainer (`explainability.py`):** This script creates the SHAP explainer that does the hard work of figuring out which words are important for each prediction.

### The Tech Stack

| Component | Technology | Why I Chose It |
| :--- | :--- | :--- |
| **Language Model** | Hugging Face Transformers (Bio_ClinicalBERT) | It's the best I could find for understanding clinical language. |
| **Classification** | Scikit-learn (Logistic Regression) | It's a solid, reliable choice that's easy to work with. |
| **Explainability** | SHAP (SHapley Additive exPlanations) | It's the gold standard for model-agnostic explainability. |
| **Dashboard** | Plotly / Dash | It's a great way to build interactive web apps with Python. |
| **Serialization** | Joblib | It's perfect for saving and loading the models and other Python objects. |

---

## Getting Started

### What You'll Need

- Python 3.8 or higher
- `venv` (trust me, you'll want to use a virtual environment)

### Installation

1.  **Clone this repo and `cd` into it.**

2.  **Set up a virtual environment.** I called mine `venv_healthai`, but you can name it whatever you want.
    ```bash
    python -m venv venv_healthai
    source venv_healthai/bin/activate
    ```
    *(On Windows, you'll need to run `venv_healthai\Scripts\activate`)*

3.  **Install the dependencies.**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Put your `mtsamples.csv` file in the `data/raw/` directory.**

### Running the Pipeline

You've got two options here:

**A) The Easy Way**

Just run the `main.py` script, and it will take care of everything.

```bash
python src/main.py
```

**B) The Step-by-Step Way**

If you want to see what each script is doing, you can run them one by one. Just make sure you do it in this order:

```bash
python src/data_preparation.py
python src/feature_extraction.py
python src/train_classifier.py
python src/explainability.py
```

### Launching the App

Once you've run the pipeline, you can start the dashboard:

```bash
python src/app.py
```
Then, open your browser and go to `http://127.0.0.1:8050/`.

---

## Local vs. Global Explanations

I've set up two different ways to look at the explanations:

-   **Local Explanations (in the app):** This is for looking at individual predictions. It answers the question: **"Why did the model classify *this specific note* this way?"** This is super useful for digging into specific cases and making sure the model is behaving as expected.

-   **Global Explanations (`visualization.py`):** This script gives you a bird's-eye view of the model's behavior. It answers the question: **"What are the most important features for each class across the entire dataset?"** This is a great way to make sure the model has learned something meaningful and isn't just picking up on noise.

---

## What's Next?

This project is just a starting point. Here are a few things I'm thinking about for the future:

1.  **Fine-tuning the BERT model:** I think I can get even better performance by fine-tuning the ClinicalBERT model on this specific dataset.
2.  **Trying out other models:** I'm curious to see how other classifiers (like XGBoost or LightGBM) would perform.
3.  **Deploying it to the cloud:** It would be cool to get this running on AWS or Google Cloud so that other people can use it.
4.  **Model distillation:** I'm also interested in seeing if I can create a smaller, faster version of the model that's still just as accurate.

---

This project uses several open-source libraries, including: pandas, numpy, scikit-learn, and joblib (BSD 3-Clause); torch (BSD); transformers, huggingface-hub, and dash-bootstrap-components (Apache 2.0); shap, plotly, and dash (MIT); and matplotlib (PSF-based).
