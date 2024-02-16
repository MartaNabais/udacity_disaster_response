import pandas as pd
from pysqlite3 import connect
from os import path


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

    Arguments:
        - cat_csv_filepath: <path to categories.csv>
        - msg_csv_filepath: <path to messages.csv>
    Output:
        - merged_df: a merged data frame from messages.csv and categories.csv
    """

    cat_df = pd.read_csv(cat_csv_filepath)
    msg_df = pd.read_csv(msg_csv_filepath)

    cat_df.loc[:, 'categories'] = cat_df.categories.str.split(';')
    cat_df = cat_df.explode('categories')
    cat_df[['categories', 'value']] = cat_df['categories'].str.split('-', expand=True)
    cat_df = cat_df.drop_duplicates(subset=['categories', 'id']).reset_index(drop=True)
    wide_cat_df = cat_df.pivot(index='id', columns='categories', values='value').reset_index(drop=True)

    # Dropping duplicates
    msg_df = msg_df.drop_duplicates().reset_index(drop=True)

    # Merge data frames
    merged_df = pd.concat([msg_df, wide_cat_df], axis=1)
    return merged_df


def load_data(merged_df, db_filepath):
    """
    This function takes the merged data frame created using extract_transform_data()
    function and loads it into a SQL database.

    Arguments:
    - merged_df: a merged data frame from messages.csv and categories.csv
    - db_filename: database filename

    Output: None.
    :rtype: object
    """
    conn = connect(db_filepath)
    table_name = path.basename(db_filepath.replace('.db', '') + '_table')
    merged_df.to_sql(table_name, con=conn, if_exists='replace')
    conn.commit()
    conn.close()
