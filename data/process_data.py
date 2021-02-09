import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Turn CSV messages and categories data into Pandas Dataframes.
    
    Input:
        messages_filepath: The path of messages dataset.
        categories_filepath: The path of categories dataset.
    Output:
        df: The merged dataset.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    '''
    Split 'categories' column in 36 categories, normalize contents,
    transform to booleans and drop duplicate rows.
    
    Input:
        df: Dataframe to be cleaned.
    Output:
        df: Cleaned dataframe.
    '''
    #select category column and splits on ';' sign into columns
    categories = df.categories.str.split(';', expand = True)
    
    #select a row
    row = categories.loc[0]  
    
    #removes last two digits of each row cell and make column name
    categories.columns = row.apply(lambda x: x[:-2])
    
    for column in categories:
        #set each value to be the last character of the string
        categories[column] = categories[column].str.get(-1)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #drop old categories and concat new
    df.drop(columns='categories', inplace=True)
    df = pd.concat([df,categories],axis=1)
    
    #drop duplicate rows
    df.drop_duplicates(inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    Save data in a database.
    
    Input:
        df: Desired dataframe to be saved.
        database_filename: Desired database filename (str).
    Output:
        None.
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(f'Loading data...\n MESSAGES PATH: {messages_filepath}\n CATEGORIES PATH: {categories_filepath}')
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print(f'Saving data...\n DATABASE: {database_filepath}')
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()