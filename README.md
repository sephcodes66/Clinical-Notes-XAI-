# Clinical-Notes-XAI: An Explainable AI Pipeline for Clinical Text Classification
This repository contains the source code for an end-to-end prototype I developed to classify clinical notes using state-of-the-art NLP. More importantly, this project was a passion project of mine to explore and demonstrate how to make these complex AI models transparent and interpretable for clinical and research applications.

*Figure 1: Interactive Dashboard ScreenShot*
![AI Metadata and Chart Output](https://github.com/sephcodes66/Clinical-Notes-XAI-/blob/main/ss/interactive_dashboard.png)

## Project Goal

In high-stakes fields like healthcare and pharmacology, the adoption of advanced AI is often limited by a "black box" problem. If researchers and clinicians cannot understand why a model makes a certain decision, they cannot fully trust or utilize it.

My primary goal with this project was to tackle this challenge directly. I aimed to build a functional tool that demonstrates how modern **Explainable AI (XAI)** techniques can be integrated with powerful deep learning models to create a system that is both **accurate** and **transparent**, aligning with the core principles of translational medicine: turning complex data science into a practical, application-oriented tool that can support clinical research and diagnostics. The final deliverable is an **interactive web dashboard** where a user can input a **clinical note**, receive a **prediction**, and immediately see which **words** and **phrases** influenced that **outcome**.

---
## Table of Contents
1. **How to Run This Project**
    1. **Project Structure**
    2. **Setup and Installation**
    3. **Running the Pipeline: Two Methods**
2. **The Technical Pipeline**
    1. **Modeling Pipeline: From Raw Text to Prediction**
    2. **Explainable AI (XAI) and Visualization**
3. **The Role of visualization.py: Global vs. Local Explanations**
4. **Future Work**

----
1. **How to Run This Project** <br>
    1. **Project Structure** <br>
        First, ensure your project directory is organized as follows:
        ```
        clinical-notes-xai/
        ├── data/
        │   ├── raw/
        │   │   └── mtsamples.csv  <-- Place the downloaded dataset here
        │   └── processed/
        ├── src/
        │   ├── phase_1_data_prep.py
        │   ├── phase_2_feature_extraction.py
        │   ├── phase_3_train_classifier.py
        │   ├── visualization.py
        │   ├── app.py
        │   └── main.py
        │
        ├── README.md
        └── requirements.txt
        ```
    2. **Setup and Installation** <br>
        - Set Up Virtual Environment:
        ```
        python -m venv venv
        ```
        
        - Activate the environment
        ```
        On macOS/Linux:
        source venv/bin/activate
        ```
        ```
        On Windows:
        venv\Scripts\activate
        ```

        - Install Requirements:
        ```
        pip install -r requirements.txt
        ```
    3. **Running the Pipeline: Two Methods**    

        Two distinct methods for executing the project pipeline, allowing for either a quick setup or a more detailed, step-by-step exploration.

        - **Method 1:** Automated Script (Recommended for Quick Setup)
        I created an automation script, **```main.py```**, to handle the core data processing and modeling tasks. It's a lean and efficient way to get the model ready.

        - From your terminal, simply run:
             ```
            python main.py
             ```
            The script will execute **data preparation, feature extraction, and model training in sequence.**

            After it completes, the pipeline correctly ends by giving you instructions on how to run the app. The final step is to launch the interactive web application:
            ```
            python app.py
            ```

        - **Method 2:** Manual Execution (Recommended for Understanding the Workflow) <br>

            If you wish to follow along with the implementation and see how each component works individually, you  can run each Python script from the project's root directory in the following precise order:
            ```
            python src/data_preparation.py
            python src/feature_extraction.py
            python src/train_classifier.py
            python src/explainability.py
            python src/visualization.py (Optional, see section below)
            python src/app.py (Launches the interactive dashboard)
            ```
---
2. **The Technical Pipeline** <br>
    1. **Modeling Pipeline**: From Raw Text to Prediction <br>
        This first stage handles the core data science task of turning unstructured clinical text into a usable prediction model.

        - **Data Preparation**: I begin by loading the raw clinical notes and applying a standard NLP preprocessing pipeline.
        - **Feature Extraction with ClinicalBERT**: To capture the deep contextual meaning of the clinical language, I use a pre-trained **```emilyalsentzer/Bio_ClinicalBERT```** model to generate a 768-dimensional vector ("embedding") that numerically represents each note's content.
        - **Classifier Training**: Using these embeddings as features, I then train a simple and fast Logistic Regression model to perform the final classification task.

    2. **Explainable AI (XAI) and Visualization** <br>
        This is the core of the project, where I make the model's decisions transparent.

        - **Implementing SHAP**: I integrated the SHAP (SHapley Additive exPlanations) library to analyze the entire pipeline and assign a precise "importance value" to every word in the text.

        - **Interactive Dashboard**: I built a web application using Dash and Plotly. This tool allows a user to input any clinical text and, in real-time, receive not only a prediction but also two powerful visualizations: a waterfall plot showing the top contributing words and highlighted text where each word is colored based on its impact.
---
3. **The Role of ```visualization.py```: Global vs. Local Explanations** <br> 

    It is important to understand the different roles of **```app.py```** and **```visualization.py```**, as they provide two different types of model explanations.

    - **```app.py``` (Local Explanation)**: The main interactive dashboard is designed to answer the question: "Why was this specific clinical note classified this way?" It provides a real-time explanation for a single data point, which is crucial for case-by-case analysis.

    - **```visualization.py``` (Global Explanation)**: This script is a separate, optional tool for deeper analysis. It answers the question: "Overall, what features does my model consider most important for a given class, across many different samples?" It does this by calculating the average importance of words over a batch of 100 notes and generating a static summary plot. This is useful for auditing the model's general behavior and ensuring it has learned clinically relevant patterns.

        In summary, **```main.py```** is designed to be a lean and efficient script to get your model ready. The **```visualization.py```** script is a separate, optional tool for deeper analysis, and **```app.py```** is the final step for user interaction.
---
4. **Future Work** <br>

    - This prototype serves as a strong foundation that I am proud of. For future work, I have identified several exciting avenues:

    - Fine-Tuning the Language Model: On a machine with a powerful GPU, I would fine-tune the ClinicalBERT model on my specific task's data to likely achieve higher accuracy.

    - Experiment with Other Classifiers: I plan to swap in models like XGBoost or a simple neural network to compare performance.

    - Deployment and Performance: I aim to deploy the Dash application to a cloud service and explore model optimization techniques like distillation to improve the real-time explanation speed.
---