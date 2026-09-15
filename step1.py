import pandas as pd


def load_csv(path):
    '''
    it reads a csv file into a pandas dataframe.
    path - its the file name or the full path of the csv.
    it returns the dataframe with all the rows and columns from the file.
    '''
    return pd.read_csv(path)


def get_columns(df):
    '''
    it splits the columns of the dataframe into numeric and non numeric ones.
    it works on any csv because the columns are found at run time and not hardcoded.
    it returns two lists - numeric_cols and categorical_cols.
    '''
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols

#test step1

# df = load_csv("iris_unlabaled.csv")

# numeric_cols, categorical_cols = get_columns(df)

# print(f"Rows: {df.shape[0]}  Columns: {df.shape[1]}")
# print(f"Numeric: {numeric_cols}")
# print(f"Categorical: {categorical_cols}")
# print(df.head(10))