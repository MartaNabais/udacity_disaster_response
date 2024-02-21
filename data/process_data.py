import sys
from os import path

import pandas as pd
from sqlalchemy import create_engine


def extract_transform_data(cat_csv_filepath, msg_csv_filepath):
    """
    This function takes two csv filepaths as input.
     The csv files were pre-processed and provided by
     1. It first reads the csv files with pandas' read.csv() function.
     2. It then processes the categories csv file by:
        - splitting the initial 'categories' column into two columns:
            one with category names (e.g., aid, water, etc.) and another with
            indicator values (0, 1)
     3. It then drops duplicated values from both data frames
     4. Finally, it merged the two data frames by making use of the 'id' column

    :param cat_csv_filepath: <path to categories.csv>
    :param msg_csv_filepath: <path to messages.csv>
    :return: merged_df
    """

    categories = pd.read_csv(cat_csv_filepath)
    messages = pd.read_csv(msg_csv_filepath)
    df = messages.merge(categories, on='id')
    categories = df.loc[:,'categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0:,]
    # use this row to extract a list of new column names for categories.
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    # Rename the columns
    categories.columns = category_colnames
    # Convert category to binary values
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    # Drop original categories values
    df.drop(columns='categories', inplace=True)
    # Concatenate the original dataframe with new categories data frame and drop duplicates
    df = pd.concat([df.copy(), categories], axis=1).drop_duplicates()

    # The 'related' column had values of 2, so I am replacing with 1
    df['related'] = df['related'].replace(2, 1)
    # Remove entries with the pattern 'NOTES' as most of these
    # contain a message from the translator saying these should be ignored
    # E.g.: 'NOTES: Regular gossip or message sharing. Not an emergency.'
    df = df[~df['message'].str.contains('NOTES')]
    return df


def load_data(merged_df, db_filepath):
    """
    This function takes the merged data frame created using extract_transform_data()
    function and loads it into a SQL database.

    :param merged_df: a merged data frame from messages.csv and categories.csv
    :param db_filepath: database file path
    :return: None.
    """
    conn = create_engine('sqlite:///' + db_filepath)
    table_name = path.basename(db_filepath.replace('.db', '') + '_table')
    merged_df.to_sql(table_name, con=conn, index=False, if_exists='replace')


def main():
    """
    Main function.
    :return:
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data & cleaning data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = extract_transform_data(messages_filepath, categories_filepath)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        load_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()