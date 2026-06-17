# /scrappers
used to scrap for students and courses data from the e-comme website.

# /data 

### The Rest of the folder
JSON: results of scrapping and filtering the data.

CSV: data used for the to train the model:

- pre-specialization -> only courses form first to levels (general only)

- all                -> all the courses across all levels (general only)

- sample             -> a few rows of the "pre-specialization" CSV files

Python: scripts to filter and transform the scrapped data.

---

# How to run the model pipeline:
### in this order: ⬇
1. 01_preprocess_and_select_features_updated.py
2. 02_train_and_compare_models_updated.py
3. 03_interactive_ui_updated.py (optional, only for testing)

## how to run '03_interactive_ui_updated.py': 
us this command: ```streamlit run 03_interactive_ui_updated.py ``` 