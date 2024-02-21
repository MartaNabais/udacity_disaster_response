# Disaster Response Pipeline Project

### Business Understanding

In the aftermath of a disaster, an overwhelming surge of communications floods various channels, including Twitter and Facebook. Unfortunately, this surge coincides with a time when disaster response organizations are least equipped to sift through and prioritize these messages. While bystanders on social media platforms rapidly disseminate information, much of it surpassing traditional news reports in speed and detail, only a fraction of these messages—roughly one in a thousand—contains actionable information for disaster response professionals.

This presents a formidable challenge in disaster management:

+ Relevant messages must be efficiently directed to the appropriate organizations responsible for different facets of disaster relief, such as providing water and medical supplies.
+ Swiftly matching the appropriate level of assistance with individuals or areas in dire need of priority attention.
  
Appen (formerly Figure Eight) has amassed a dataset comprising over 30,000 authentic messages dispatched to disaster response entities during significant calamities, including the 2010 Haiti earthquake, the Chile earthquake of the same year, the 2010 floods in Pakistan, Superstorm Sandy in 2012, and numerous other incidents spanning more than a hundred different disasters. These messages have been amalgamated, refined, and standardized across various disasters, enabling comprehensive analysis of distinct patterns and the development of supervised machine learning models.

In this project, which is part of the Udacity's Data Science Nanodegree. I have analyzed disaster data from Appen to build a model for an API that classifies disaster messages, that aimds to address the issues mentioned above.

### Technical details - or how to run the code
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
        - `python process_data.py categories.csv messages.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it in a pickle file
        - `cd models`
        - `python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

3. Go to `app` directory: `cd app`

4. Run your web app: `python run.py`

5. Check the webapp from an internet browser: ```http://127.0.0.1:3000```

Note: if you don't want to run the full pipeline, you can run the app code
directly, but first you need to unzip the pickle file ```models/classifier.pkl.gz```

