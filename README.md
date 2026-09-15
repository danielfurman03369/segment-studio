# Segment Studio

An interactive Streamlit app that finds hidden segments in any uploaded CSV using K-Means. It then uses an LLM to give each cluster a short name and a one-line description.

```
CSV -> preprocessing & scaling -> WCSS/elbow -> K-Means -> LLM naming -> CSV
```

## The five steps

**1. Upload & choose columns.** The user uploads a CSV through `st.file_uploader`. `load_csv` reads it and `get_columns` splits the columns into numeric and categorical at runtime. Both lists are shown as multiselects, defaulting to "everything," so the user can untick any column that shouldn't be used for clustering.

**2. Find k with the elbow plot.** The user picks a Min k / Max k range. `scale_data` preprocesses the selected columns (median-fills numeric gaps, fills categorical gaps with `"missing"`, one-hot encodes categoricals, then applies `StandardScaler`), and `compute_wcss` runs K-Means once for every k in the range, recording the inertia (WCSS) each time. `plot_elbow` draws the resulting elbow chart.

**3. Create the clusters.** The user picks a single k, bounded to whatever range was scanned in step 2. The slider defaults to a simple elbow estimate the app computes from the WCSS curve, but any k in range can be chosen. `create_clusters` runs K-Means once more with that k, and `cluster_counts` builds a table of row counts per cluster.

**4. Name the clusters with AI.** `summarize_clusters` builds a compact per-cluster summary (row count, column averages, most common categorical values) instead of sending raw rows. `name_clusters` turns that into a prompt via `build_prompt`, sends it to the LLM through `ask_llm`, and parses the reply back into a name and description per cluster.

**5. Export.** The app builds `cluster_id` and `cluster_name` columns the same way `step5.export_clustered` does, but keeps the result in memory rather than importing and calling that function. Nothing is written to disk until the user presses "Download clustered CSV" — `st.download_button` streams the bytes at that point only.

## Project structure

```
segment-studio/
├── app.py               Streamlit UI. The only file that imports streamlit;
│                        wires steps 1-5 together with st.session_state.
├── step1.py             load_csv, get_columns - read the CSV, split columns
│                        into numeric/categorical.
├── step2.py             scale_data, compute_wcss, plot_elbow - preprocessing,
│                        scaling, and the WCSS/elbow calculation.
├── step3.py             create_clusters, cluster_counts - run K-Means, count
│                        rows per cluster.
├── step4.py             summarize_clusters, build_prompt, name_clusters -
│                        build cluster summaries and ask the LLM to name them.
├── step5.py             export_clustered - adds cluster_id/cluster_name
│                        columns and writes the exported CSV. Not imported by
│                        app.py (see Step 5 above); kept as the standalone
│                        logic module.
├── llm_func.py          ask_llm - sends a prompt to the Ollama API and
│                        returns the model's reply.
├── api_key.py           MY_API_KEY - your Ollama API key. You create this
│                        file yourself (see Setup); it is gitignored.
├── iris_unlabaled.csv   Sample dataset. Contains an index column and a label
│                        column - see Notes below.
├── requirements.txt
└── .gitignore
```

`step1.py` through `step5.py` hold all the actual logic (CSV loading, scaling, K-Means, summarization, export) and contain no Streamlit code. `app.py` is UI only - it imports and calls those functions and never reimplements them.

## Setup

```
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # macOS/Linux

pip install -r requirements.txt
```

Create `api_key.py` in the project root:

```python
MY_API_KEY = "your key"
```

`api_key.example.py` is included as a template for this - copy it to `api_key.py` and fill in your own key.

Run the app:

```
streamlit run app.py
```

If `streamlit` isn't on your PATH, fall back to:

```
py -m streamlit run app.py
```

`api_key.py` is listed in `.gitignore` and must be created locally - it is not committed. Without it, steps 1-3 still work; step 4 shows an error explaining that `api_key.py` with `MY_API_KEY` is required.

## Notes

- The app works on any CSV, not just the sample one: columns are detected at runtime by `get_columns`, nothing is hardcoded, and the user chooses which numeric and categorical columns to actually cluster on.
- ID columns and label columns are numeric but meaningless or circular as clustering features, and should be unticked in step 1. The included `iris_unlabaled.csv` has both: `Unnamed: 0` (a row index) and `target` (the true species label). Leaving either ticked lets the "clusters" just reconstruct the index or the label instead of discovering structure.
- The LLM is called through the Ollama API (`https://ollama.com/api/chat`). The model name (`gpt-oss:120b`) is set in `llm_func.py`.

## Possible improvements

Not implemented, but reasonable next steps:

- Silhouette score as an alternative or addition to WCSS for evaluating k.
- A more rigorous automatic k selection method (e.g. picking k by silhouette score) - currently the app only suggests a default k from the elbow curve, which the user can override.
- Outlier removal before clustering.
