import os

import pandas as pd
import streamlit as st

from step1 import load_csv, get_columns
from step2 import scale_data, compute_wcss, plot_elbow
from step3 import create_clusters, cluster_counts

try:
    from step4 import summarize_clusters, name_clusters
    from api_key import MY_API_KEY
    if not MY_API_KEY:
        raise ImportError("MY_API_KEY is empty")
    llm_available = True
except ImportError:
    summarize_clusters, name_clusters = None, None
    llm_available = False


def find_elbow_k(scores):
    """Estimate the elbow point of a k/wcss curve using the max-distance-to-chord method."""
    ks = scores["k"].to_numpy()
    wcss = scores["wcss"].to_numpy()
    if len(ks) < 3:
        return int(ks[0])

    x_range = ks.max() - ks.min()
    y_range = wcss.max() - wcss.min()
    x = (ks - ks.min()) / x_range if x_range else ks * 0.0
    y = (wcss - wcss.min()) / y_range if y_range else wcss * 0.0

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]
    denom = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
    if denom == 0:
        return int(ks[0])

    distances = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denom
    return int(ks[int(distances.argmax())])


st.set_page_config(page_title="Segment Studio", page_icon="\U0001F9E9", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }
    h1 { font-weight: 700; }
    h2 { font-weight: 600; margin-top: 0.25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("\U0001F9E9 Segment Studio")
st.caption("Upload a CSV, find natural clusters with K-Means, and let an LLM name them.")

metric_cols = st.columns(4)
metric_cols[0].metric("Rows", st.session_state["df"].shape[0] if "df" in st.session_state else "–")
metric_cols[1].metric("Columns", st.session_state["df"].shape[1] if "df" in st.session_state else "–")
metric_cols[2].metric("Chosen k", st.session_state.get("k_chosen", "–"))
metric_cols[3].metric(
    "Clusters",
    len(st.session_state["cluster_table"]) if "cluster_table" in st.session_state else "–",
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1: upload and column selection
# ---------------------------------------------------------------------------
st.header("1. Upload & choose columns")

uploaded_file = st.file_uploader("CSV file", type="csv")

if uploaded_file is not None and st.session_state.get("uploaded_name") != uploaded_file.name:
    df = load_csv(uploaded_file)
    numeric_cols_all, categorical_cols_all = get_columns(df)

    st.session_state["df"] = df
    st.session_state["original_filename"] = uploaded_file.name
    st.session_state["uploaded_name"] = uploaded_file.name
    st.session_state["numeric_cols_all"] = numeric_cols_all
    st.session_state["categorical_cols_all"] = categorical_cols_all

    for key in (
        "numeric_cols_ms", "categorical_cols_ms",
        "X", "wcss_table", "wcss_fig",
        "min_k", "max_k", "wcss_min_k", "wcss_max_k",
        "labels", "k_chosen", "k_slider", "cluster_table",
        "summaries", "names", "descriptions",
        "export_path", "export_df",
    ):
        st.session_state.pop(key, None)

    st.rerun()

if "df" in st.session_state:
    df = st.session_state["df"]

    st.dataframe(df, use_container_width=True)

    col_a, col_b = st.columns(2)
    numeric_cols_selected = col_a.multiselect(
        "Numeric columns to cluster on",
        options=st.session_state["numeric_cols_all"],
        default=st.session_state["numeric_cols_all"],
        key="numeric_cols_ms",
    )
    categorical_cols_selected = col_b.multiselect(
        "Categorical columns to cluster on",
        options=st.session_state["categorical_cols_all"],
        default=st.session_state["categorical_cols_all"],
        key="categorical_cols_ms",
    )
else:
    st.info("Upload a CSV to get started.")
    numeric_cols_selected, categorical_cols_selected = [], []

st.divider()

# ---------------------------------------------------------------------------
# Step 2: WCSS and elbow plot
# ---------------------------------------------------------------------------
st.header("2. Find k with the elbow plot")

step2_ready = "df" in st.session_state and (numeric_cols_selected or categorical_cols_selected)

if not step2_ready and "df" in st.session_state:
    st.warning("Pick at least one column in step 1 before computing WCSS.")

if "df" in st.session_state:
    k_cap = max(2, min(20, len(st.session_state["df"]) - 1))
else:
    k_cap = 20

range_col1, range_col2 = st.columns(2)
min_k = range_col1.slider(
    "Min k",
    min_value=2,
    max_value=k_cap,
    value=2,
    disabled=not step2_ready,
    key="min_k",
)
max_k = range_col2.slider(
    "Max k",
    min_value=2,
    max_value=k_cap,
    value=min(10, k_cap),
    disabled=not step2_ready,
    key="max_k",
)

range_invalid = step2_ready and min_k > max_k
if range_invalid:
    st.warning("Min k must be less than or equal to Max k.")

run_wcss = st.button("Run WCSS", disabled=not step2_ready or range_invalid)

if run_wcss:
    X = scale_data(st.session_state["df"], numeric_cols_selected, categorical_cols_selected)
    scores = compute_wcss(X, min_k, max_k)
    fig = plot_elbow(scores)

    st.session_state["X"] = X
    st.session_state["wcss_table"] = scores
    st.session_state["wcss_fig"] = fig
    st.session_state["wcss_min_k"] = min_k
    st.session_state["wcss_max_k"] = max_k

    for key in ("labels", "k_chosen", "cluster_table", "summaries", "names", "descriptions", "export_path", "export_df", "k_slider"):
        st.session_state.pop(key, None)

if "wcss_table" in st.session_state:
    col_e, col_f = st.columns([1, 2])
    col_e.dataframe(st.session_state["wcss_table"], use_container_width=True)
    col_f.pyplot(st.session_state["wcss_fig"])

st.divider()

# ---------------------------------------------------------------------------
# Step 3: create clusters
# ---------------------------------------------------------------------------
st.header("3. Create the clusters")

step3_ready = "X" in st.session_state

if step3_ready:
    k_min_bound = st.session_state["wcss_min_k"]
    k_max_bound = st.session_state["wcss_max_k"]
    k_default = find_elbow_k(st.session_state["wcss_table"])
else:
    k_min_bound, k_max_bound, k_default = 2, k_cap, 3

k_col, k_info_col = st.columns([2, 1])
k = k_col.slider(
    "k (number of clusters)",
    min_value=k_min_bound,
    max_value=k_max_bound,
    value=k_default,
    disabled=not step3_ready,
    key="k_slider",
)
k_info_col.metric("Selected k", k)

create_clicked = st.button("Create clusters", disabled=not step3_ready)

if create_clicked:
    labels = create_clusters(st.session_state["X"], k)
    table = cluster_counts(labels)

    st.session_state["labels"] = labels
    st.session_state["k_chosen"] = k
    st.session_state["cluster_table"] = table

    for key in ("summaries", "names", "descriptions", "export_path", "export_df"):
        st.session_state.pop(key, None)

if "cluster_table" in st.session_state:
    st.dataframe(st.session_state["cluster_table"], use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Step 4: name the clusters with the LLM
# ---------------------------------------------------------------------------
st.header("4. Name the clusters with AI")

step4_ready = "labels" in st.session_state

if not llm_available:
    st.error(
        "AI cluster naming is unavailable: create api_key.py in the project "
        "folder with MY_API_KEY set to your Ollama API key."
    )

name_clicked = st.button(
    "Name clusters with AI", disabled=not step4_ready or not llm_available
)

if name_clicked:
    try:
        with st.spinner("Asking the LLM to name the clusters..."):
            summaries = summarize_clusters(
                st.session_state["df"],
                st.session_state["labels"],
                numeric_cols_selected,
                categorical_cols_selected,
            )
            names, descriptions = name_clusters(summaries)

        table = st.session_state["cluster_table"].copy()
        table["name"] = table["cluster_id"].map(names)
        table["description"] = table["cluster_id"].map(descriptions)

        st.session_state["summaries"] = summaries
        st.session_state["names"] = names
        st.session_state["descriptions"] = descriptions
        st.session_state["cluster_table"] = table

        for key in ("export_path", "export_df"):
            st.session_state.pop(key, None)
    except Exception as exc:
        st.error(f"Could not name the clusters: {exc}")

if "summaries" in st.session_state:
    st.dataframe(st.session_state["cluster_table"], use_container_width=True)
    with st.expander("What was sent to the LLM"):
        st.json(st.session_state["summaries"])

st.divider()

# ---------------------------------------------------------------------------
# Step 5: export
# ---------------------------------------------------------------------------
st.header("5. Export")

step5_ready = "names" in st.session_state

if step5_ready and "export_df" not in st.session_state:
    export_df = st.session_state["df"].copy()
    export_df["cluster_id"] = st.session_state["labels"]
    export_df["cluster_name"] = (
        pd.Series(st.session_state["labels"])
        .map(st.session_state["names"]).values
    )
    st.session_state["export_df"] = export_df
    st.session_state["export_path"] = (
        st.session_state["original_filename"].replace(".csv", "_clustered.csv")
    )

if "export_df" in st.session_state:
    st.dataframe(st.session_state["export_df"], use_container_width=True)
    csv_bytes = st.session_state["export_df"].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download clustered CSV",
        data=csv_bytes,
        file_name=os.path.basename(st.session_state["export_path"]),
        mime="text/csv",
        disabled=not step5_ready,
    )
else:
    st.button("Download clustered CSV", disabled=True)
    st.caption("Name the clusters in step 4 first.")
