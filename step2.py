import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from step1 import load_csv, get_columns


def scale_data(df, numeric_cols, categorical_cols):
    '''
    it prepares the data for k-means.
    the missing numbers are filled with the median of the column, missing text with "missing".
    the text columns are turned into 0/1 columns with get_dummies so k-means can use them.
    all the columns are then scaled with standardscaler
    then it returns a numpy array of the scaled data.
    '''

    numeric = df[numeric_cols].fillna(df[numeric_cols].median())

    if len(categorical_cols) > 0:
        categorical = df[categorical_cols].fillna("missing")
        dummies = pd.get_dummies(categorical)
        data = pd.concat([numeric, dummies], axis=1)
    else:
        data = numeric

    scaler = StandardScaler()
    return scaler.fit_transform(data)


def compute_wcss(X, min_k, max_k):
    '''
    it runs k-means once for every k between min_k and max_k.
    for each k it saves the wcss, which sklearn
    already calculates and keeps in model.inertia_.
    and then it returns a dataframe with two columns - k and wcss.
    '''
    k_values = []
    wcss_values = []

    for k in range(min_k, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(X)
        k_values.append(k)
        wcss_values.append(model.inertia_)

    return pd.DataFrame({"k": k_values, "wcss": wcss_values})


def plot_elbow(scores):
    '''
    it draws the elbow graph - k on the x axis and wcss on the y axis.
    it returns the matplotlib figure so it can be shown in the terminal.
    '''
    fig, ax = plt.subplots()
    ax.plot(scores["k"], scores["wcss"], marker="o")
    ax.set_xlabel("k")
    ax.set_ylabel("WCSS (inertia)")
    ax.set_title("Elbow Plot")
    return fig


#test step2

# df = load_csv("iris_unlabaled.csv")
# numeric_cols, categorical_cols = get_columns(df)

# X = scale_data(df, numeric_cols, categorical_cols)
# print("Scaled shape:", X.shape)

# scores = compute_wcss(X, 2, 10)
# print(scores)

# fig = plot_elbow(scores)
# plt.show()
