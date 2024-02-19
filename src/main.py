import logging
import os
import sys
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('path_cat_csv',
                        type=str, help='Filepath to categories.csv')
    parser.add_argument('path_msg_csv',
                        type=str, help='Filepath to messages.csv')
    parser.add_argument('path_db',
                        type=str,  help='Filepath to database.db')
    parser.add_argument('model_filepath',
                        type=str,  help='Filepath to model pickle file')
    args = parser.parse_args()

    logger.info("Starting Pipeline!")
    logger.info("Extracting and Transforming Data")
    merged_df = data.process_data.extract_transform_data(args.path_cat_csv, args.path_msg_csv)
    logger.info("Loading merged data frame into db...")
    data.process_data.load_data(merged_df, args.path_db)
    logger.info("ETL Pipeline Finished!")
    logger.info("Training NLP-ML Pipeline...")
    X_train, X_test, y_train, y_test, labels = models.train_classifier.split_train_test(args.path_db)
    model = models.train_classifier.build_model_pipeline()
    model.fit(X_train, y_train)
    logger.info("NLP-ML Pipeline Finished!")
    logger.info("Results of predicting in Test Data:")
    y_pred = model.predict(X_test)
    models.train_classifier.evaluate_model(model, y_test, y_pred, labels)
    logger.info("Saving model in pickle file...")
    models.train_classifier.save_model(model, args.model_filepath)
    logger.info("End of Pipeline!")


if __name__ == "__main__":
    # Always better to write relative path
    # than absolute path
    # and always best to use the path functions
    # instead of strings
    root = os.getcwd()
    sys.path.append(root)
    sys.path.append(os.path.join(root, "src"))

    # Logging
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s\n%(asctime)s.%(msecs).03d',
                        datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(__name__)

    # Functions
    import data.process_data
    import models.train_classifier

    main()
