# Disaster Response Pipeline Project

### Summary

This project is part of the Udacity's Data Science Nanodegree.
I have analyzed disaster data from Appen (formerly Figure 8) to
build a model for an API that classifies disaster messages.
Briefly:
1. The ETL (Extract-Transform-Load) Pipeline is located in `data/process_data.py` which:
    + Loads the `messages` and `categories` datasets
    + Merges the two datasets
    + Cleans the data
    + Stores it in a SQLite database


2. The ML (Machine Learning) Pipeline is located in `models/train_classifier.py` which:
    + Loads data from the SQLite database
    + Splits the dataset into training and test sets
    + Builds a text processing and machine learning pipeline
    + Trains and tunes a model using GridSearchCV
    + Outputs results on the test set
    + Exports the final model as a pickle file


3. Finally, the project also provides a Flask web app where an emergency worker can input a new message and get
   classification results in several categories. The web app alsos displays visualizations of the data.

### File Structure
```
 - app
  | - template
  | |- master.html  # main page of web app
  | |- go.html  # classification result page of web app
  |- run.py  # Flask file that runs app

- data
  |- categories.csv  # data to process
  |- messages.csv  # data to process
  |- process_data.py
  |- DisasterResponse.db   # database to save clean data to

- models
  |- train_classifier.py
  |- classifier.pkl  # saved model
 
- environment.yml # yaml file with environment configuration specs

- requirements.txt # text file with python packages required to run the module

- README.md: readme file.
```

### Instructions:
1. First install conda or miniconda, if not installed yet. [See here.](https://docs.anaconda.com/free/miniconda/miniconda-install/)
2. Clone this git repository: ```git clone https://github.com/MartaNabais/udacity_disaster_response.git```
3. Run the following command in the project's root directory:
    - `conda env create --file environment.yml -p ~/anaconda3/envs/udacity_disaster_response` **or**
    - `conda env create --file environment.yml -p ~/miniconda3/envs/udacity_disaster_response`
3. Run the following commands in the project's root directory to set up your database and model.
    - To load environment:
        - `conda activate ~/anaconda3/env/udacity_disaster_response` **or**
        - `conda activate ~/miniconda3/env/udacity_disaster_response`
    - To run ETL that cleans data and stores in database:
        - `cd data`
        - `python process_data.py disaster_categories.csv disaster_messages.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it in a pickle file
        - `cd models`
        - `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Check the webapp from an internet browser: ```http://127.0.0.1:3000```

Note: if you don't want to run the full pipeline, you can run the app code
directly, but first you need to unzip the pickle file ```models/classifier.pkl.gz```
