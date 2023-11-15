import pandas as pd
import numpy as np
import os
from env import get_db_url
from sklearn.model_selection import train_test_split


def get_zillow_2017():
    '''This function imports zillow 2017 data from MySql codeup server and creates a csv
    
    argument: df
    
    returns: zillow df'''
    filename = "zillow_2017.csv"
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        query = """
        SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
        FROM properties_2017
        JOIN propertylandusetype
        USING (propertylandusetypeid)
        WHERE propertylandusetypeid like '261';"""
        connection = get_db_url("zillow")
        df = pd.read_sql(query, connection)
        df.to_csv(filename, index=False)
    return df

def wrangle_zillow():
    ''' This function takes the zillow df csv, cleans it, and returns prep'd df'''

    file = 'zillow_2017.csv'

    df = pd.read_csv(file)# reads csv


    df = df.dropna() #drops all NaN values

    # renames columns for ease
    df.rename(columns={df.columns[0]: 'bedrooms', df.columns[1]: 'bathrooms', df.columns[2]: 'finished_area', df.columns[3]: 'home_value', df.columns[4]: 'year_built', df.columns[5]: 'tax_amount', df.columns[6]: 'county'}, inplace=True)

    # turn into int
    make_ints = ['bedrooms', 'finished_area', 'home_value', 'year_built']
    for col in make_ints:
        df[col] = df[col].astype(int)
    # change county values
    df.county = df.county.map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})

    return df

def check_columns(df):
    """
    This function takes a pandas dataframe as input and returns
    a dataframe with information about each column in the dataframe. For
    each column, it returns the column name, the number of
    unique values in the column, the unique values themselves,
    the number of null values in the column, and the data type of the column.
    The resulting dataframe is sorted by the 'Number of Unique Values' column in ascending order.
    - pandas dataframe
    """
    data = []
    # Loop through each column in the dataframe
    for column in df.columns:
        # Append the column name, number of unique values, unique values, number of null values, and data type to the data list
        data.append(
            [
                column,
                df[column].nunique(),
                df[column].unique(),
                df[column].isna().sum(),
                df[column].isna().mean(),
                df[column].dtype
            ]
        )
    # Create a pandas dataframe from the data list, with column names 'Column Name', 'Number of Unique Values', 'Unique Values', 'Number of Null Values', and 'dtype'
    # Sort the resulting dataframe by the 'Number of Unique Values' column in ascending order
    return pd.DataFrame(
        data,
        columns=[
            "Column Name",
            "Number of Unique Values",
            "Unique Values",
            "Number of Null Values",
            "Proportion of Null Values",
            "dtype"
        ],
    ).sort_values(by="Number of Unique Values")

def split_continuous(df):
    '''
    split continuouse data into train, validate, test; No target variable

    argument: df

    return: train, validate, test
    '''

    train_val, test = train_test_split(df,
                                   train_size=0.8,
                                   random_state=1108,
                                   )
    train, validate = train_test_split(train_val,
                                   train_size=0.7,
                                   random_state=1108,
                                   )
    
    print(f'Train: {len(train)/len(df)}')
    print(f'Validate: {len(validate)/len(df)}')
    print(f'Test: {len(test)/len(df)}')
    

    return train, validate, test

def next_split(train, validate, test):
    '''This function creates your modeling variables with the train, validate, test 
    sets and returns them from the zillow set
    
    argument: train, validate, test
    
    return: X_train, X_validate, X_test, y_train, y_validate, y_test'''

    X_train = train

    X_validate = validate

    X_test = test

    y_train = train['county']

    y_validate = validate['county'] 

    y_test = test['county']

    return X_train, X_validate, X_test, y_train, y_validate, y_test