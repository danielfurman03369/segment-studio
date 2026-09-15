import pandas as pd

from step1 import load_csv, get_columns
from step2 import scale_data
from step3 import create_clusters, cluster_counts
from step4 import summarize_clusters, name_clusters


def export_clustered(df, labels, names, original_path):
    '''
    it creates a new version of the original data with the clusters added.
    adds the columns cluster_id and cluster_name, where cluster_name is the name
    the llm gave to the cluster of that row.
    then the file is saved as <original_name>_clustered.csv with index=false so there is no extra
    index column is added.
    and then itreturns the new file name and the new dataframe.
    '''
    result = df.copy()

    result["cluster_id"] = labels
    result["cluster_name"] = pd.Series(labels).map(names).values

    new_path = original_path.replace(".csv", "_clustered.csv")
    result.to_csv(new_path, index=False)

    return new_path, result


# test run of the whole project code (steps 1 to 5)
# the ui will replace the hardcoded parts: the file name comes from the
# file uploader, the dropped columns from the user's column selection,
# and the k from the slider

# csv_path = "iris_unlabaled.csv"

# df = load_csv(csv_path)
# df = df.drop(columns=["Unnamed: 0", "target"])   # testing only - remove later

# numeric_cols, categorical_cols = get_columns(df)
# X = scale_data(df, numeric_cols, categorical_cols)
# labels = create_clusters(X, 3)

# summaries = summarize_clusters(df, labels, numeric_cols, categorical_cols)
# names, descriptions = name_clusters(summaries)

# new_path, result = export_clustered(df, labels, names, csv_path)

# print("Saved:", new_path)
# print(result.head(10))