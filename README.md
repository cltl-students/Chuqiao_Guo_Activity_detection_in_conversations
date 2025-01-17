# Chuqiao_Guo_Activity_detection_in_conversations
**Master's Degree in Linguistics: Text Mining, Vrije Universiteit Amsterdam 2023/2024**


This repository provides the codes and datasets used in the thesis: Extracting Activity Information with LLMs Using GPT-Generated Data

## Contents

### 1. Folder: code
- Jupyter Notebook: step1_gpt_generated_conversations.ipynb
  - This notebook generates natural conversations between a patient and a healthcare chatbot. In the conversations, the patient describes their daily activities to the chatbot, and the chatbot asks further questions for more detailed information.

- Jupyter Notebook: step2_data_cleaning.ipynb
  - This notebook formats the generated data and calculates the statistics of the dataset

- Jupyter Notebook: step3_resizing_files.ipynb
    - This notebook resizes each category of generated data into four smaller files for better processing by GPT
    - It also includes the train-dev-test split process

- Jupyter Notebook: step4a_prepare_for_annotation.ipynb
    - In this notebook, the dev and test datasets are pre-processed to a format that accommodates manual annotation.
    - After pre-processing, the dataset will have 8 columns, namely: 'conversation', 'sent_id', 'token_id', 'token', 'event', 'time', 'place', 'participant'.
    - The annotators can mark the 'event', 'time', 'place', 'participant' information to annotate the event-related information.

- Jupyter Notebook: step4b_annotation_to_bio_labels.ipynb
    - This Notebook converts the manually annotated file "annotation_test.tsv" and "annotation_dev.tsv" to a ready-to-evaluation format.
    - The output file only contains the conversation id, sentence id, token id, token, and the BIO system labels.

- Jupyter Notebook: step5a_gpt_predict_labels.ipynb
  - This notebook includes the prompt engineering process of predicting event-related information using GPT-4o

- Jupyter Notebook: step5b_label_distribution.ipynb
  - This notebook shows the statistics of the generated dataset. Please run it after running "step5_gpt_predicted_labels" to see the distributions of generated labels.

- Jupyter Notebook: step6a_fine-tune_bert.ipynb
    - This notebook fine-tunes the multilingual BERT model.
    - For a higher speed of processing, it is suggested to run this notebook in Google Colab.

- Jupyter Notebook: step6b_prediction_and_evaluation.ipynb
    - This notebook tests the model performance of the fine-tuned BERT model, and generates a file integrated into the dev/test dataset and the corresponding predictions.
    - For a higher speed of processing, it is suggested to run this notebook in Google Colab.

- Jupyter Notebook: step7_rule_based_system.ipynb
    - This notebook includes the process of defining rules to detect event-related tokens using the dependency parser and NER processor of SpaCy.
    - It outputs the predictions of the rule-based system in this dir: 'response_data/dataset/rule-based/'

- Jupyter Notebook: step8_optimisation_and_eveluation.ipynb
  
  - This notebook elaborates on the optimisation process (using dev set)
  - Also the final evaluation process (using a test set)
  
- Python File: utils_srl.py
   - The utils file used to fine-tune and evaluate the BERT-based model

- Python File: utils.py
  - The utils file used in other modules.

### Folder: response_data
This folder includes the dataset used in the experiment.

### Folder: Fig
This folder contains the confusion matrices of the systems

### TXT File: requirements.txt
Includes all packages used in the project. Can be used to install the dependencies

### Thesis Report: Chuqiao_Guo_MA_Thesis.pdf
The pdf which contains the full thesis report

## References
The code for generating the data through prompt engineering and part of the code for data cleaning is inspired by [ICF-activities-classifier](https://github.com/cltl-students/ICF-activities-classifier)

The code for fine-tuning the multilingual BERT model is adopted from [bert4srl](https://github.com/angel-daza/bert4srl)
