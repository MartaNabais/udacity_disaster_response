import logging
import os
import sys
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('path_cat_csv', type=str, help='Filepath to categories.csv')
    parser.add_argument('path_msg_csv', type=str, help='Filepath to messages.csv')
    parser.add_argument('path_db', type=str,  help='Filepath to database.db')
    args = parser.parse_args()
    logger.info("Extracting and Transforming Data")
    merged_df = data.process_data.extract_transform_data(args.path_cat_csv, args.path_msg_csv)
    logger.info("Loading merged data frame into db...")
    data.process_data.load_data(merged_df, args.path_db)
    logger.info("Done!")


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

    # Classes
    import data.process_data

    main()
