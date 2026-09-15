import pandas as pd
from sklearn.cluster import KMeans

from step1 import load_csv, get_columns
from step2 import scale_data


def create_clusters(X, k):
    '''
    it runs k-means with the k that the user chose.
    it returns model.labels_ - an array with one cluster number for every row,
    in the same order as the rows in the df.
    '''
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)
    return model.labels_


def cluster_counts(labels):
    '''
    it counts how many rows are in every cluster.
    then it returns a dataframe with the columns cluster_id, count, name and description.
    name and description are empty here and will be filled in step 4.
    '''
    counts = pd.Series(labels).value_counts().sort_index()

    table = pd.DataFrame({
        "cluster_id": counts.index,
        "count": counts.values,
        "name": "",
        "description": ""
    })

    return table

# test step3

# df = load_csv("iris_unlabaled.csv")
# numeric_cols, categorical_cols = get_columns(df)
# X = scale_data(df, numeric_cols, categorical_cols)

# labels = create_clusters(X, 3)

# print(cluster_counts(labels))