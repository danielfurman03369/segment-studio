import pandas as pd

from step1 import load_csv, get_columns
from step2 import scale_data
from step3 import create_clusters, cluster_counts
from llm_func import ask_llm


def summarize_clusters(df, labels, numeric_cols, categorical_cols):
    '''
    it builds a short summary for every cluster - the number of rows, the average of
    every numeric column, and the most common values of the text columns.
    the full rows are not sent to the llm because there are too many of them,
    the averages are what shows the difference between the clusters.
    it returns a list of dictionaries, one for every cluster.
    '''
    summaries = []

    for cluster_id in sorted(set(labels)):
        rows = df[labels == cluster_id]

        info = {}
        info["cluster_id"] = int(cluster_id)
        info["count"] = int(len(rows))

        averages = {}
        for col in numeric_cols:
            averages[col] = round(float(rows[col].mean()), 2)
        info["averages"] = averages

        top_values = {}
        for col in categorical_cols:
            top_values[col] = rows[col].value_counts().head(3).to_dict()
        if len(top_values) > 0:
            info["most_common"] = top_values

        summaries.append(info)

    return summaries


def build_prompt(summaries):
    '''
    it turns the cluster summaries into the text that is sent to the llm.
    the prompt asks for one line per cluster in the format:
    cluster_id | name | description
    so the answer can be split later with split.
    then it returns the prompt as a string.
    '''
    prompt = "Here are clusters found by K-Means:\n\n"

    for info in summaries:
        prompt = prompt + "Cluster " + str(info["cluster_id"])
        prompt = prompt + " (" + str(info["count"]) + " rows): "
        prompt = prompt + str(info["averages"]) + "\n"

    prompt = prompt + "\nFor each cluster give a short name and a one sentence description."
    prompt = prompt + "\nAnswer with one line per cluster, exactly like this, nothing else:"
    prompt = prompt + "\n0 | Short Name | One sentence description."
    return prompt


def name_clusters(summaries):
    '''
    it sends the prompt to the llm and reads the answer.
    every line of the answer is split by "|" - lines that do not have 3 parts are skipped.
    it returns two dictionaries - names and descriptions, both by cluster_id.
    '''
    answer = ask_llm(build_prompt(summaries))

    names = {}
    descriptions = {}

    for line in answer.split("\n"):
        parts = line.split("|")
        if len(parts) == 3:
            cluster_id = int(parts[0].strip())
            names[cluster_id] = parts[1].strip()
            descriptions[cluster_id] = parts[2].strip()

    return names, descriptions


# test step4

# df = load_csv("iris_unlabaled.csv")
# numeric_cols, categorical_cols = get_columns(df)
# X = scale_data(df, numeric_cols, categorical_cols)
# labels = create_clusters(X, 3)

# summaries = summarize_clusters(df, labels, numeric_cols, categorical_cols)
# print(summaries)

# names, descriptions = name_clusters(summaries)

# table = cluster_counts(labels)
# table["name"] = table["cluster_id"].map(names)
# table["description"] = table["cluster_id"].map(descriptions)

# print(table)