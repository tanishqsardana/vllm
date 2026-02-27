from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Metrics Comparison Dashboard", layout="wide")

NUMERIC_AGGREGATIONS: dict[str, Callable[[pd.Series], float]] = {
    "mean": lambda s: float(s.mean()),
    "median": lambda s: float(s.median()),
    "min": lambda s: float(s.min()),
    "max": lambda s: float(s.max()),
    "sum": lambda s: float(s.sum()),
    "p90": lambda s: float(s.quantile(0.90)),
    "p95": lambda s: float(s.quantile(0.95)),
    "p99": lambda s: float(s.quantile(0.99)),
    "count": lambda s: float(s.count()),
}


@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def list_run_dirs(results_root: Path) -> list[str]:
    if not results_root.exists() or not results_root.is_dir():
        return []
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def common_csv_files(results_root: Path, run_names: list[str]) -> list[str]:
    common: set[str] | None = None
    for run in run_names:
        files = {p.name for p in (results_root / run).glob("*.csv")}
        if common is None:
            common = files
        else:
            common &= files
    return sorted(common or [])


def main() -> None:
    st.title("Metrics Comparison Dashboard")
    st.caption(
        "Choose a CSV and directly compare row-level values across run folders."
    )

    with st.sidebar:
        st.header("Inputs")
        results_dir = st.text_input("Results directory", value="results")

    results_root = Path(results_dir).expanduser().resolve()
    run_names = list_run_dirs(results_root)
    if not run_names:
        st.error(f"No run folders found in: {results_root}")
        st.stop()

    default_runs = run_names[-min(len(run_names), 4) :]
    with st.sidebar:
        selected_runs = st.multiselect("Runs to compare", run_names, default=default_runs)
    if not selected_runs:
        st.warning("Select at least one run.")
        st.stop()

    files = common_csv_files(results_root, selected_runs)
    if not files:
        st.error("No common CSV files found across selected runs.")
        st.stop()

    default_file = "requests_soak_5_constant.csv" if "requests_soak_5_constant.csv" in files else files[0]
    with st.sidebar:
        selected_file = st.selectbox("CSV file", files, index=files.index(default_file))
        max_rows_for_chart = st.slider("Max bars shown", min_value=50, max_value=2000, value=300, step=50)

    df_by_run: dict[str, pd.DataFrame] = {}
    for run in selected_runs:
        path = results_root / run / selected_file
        try:
            df_by_run[run] = read_csv_cached(str(path))
        except Exception as exc:
            st.error(f"Failed to read {path}: {exc}")
            st.stop()

    shared_columns: set[str] | None = None
    for df in df_by_run.values():
        cols = set(df.columns)
        if shared_columns is None:
            shared_columns = cols
        else:
            shared_columns &= cols
    ordered_shared_columns = [col for col in df_by_run[selected_runs[0]].columns if col in (shared_columns or set())]

    if not ordered_shared_columns:
        st.error("No shared columns found for this CSV across selected runs.")
        st.stop()

    default_column = "latency_ms" if "latency_ms" in ordered_shared_columns else ordered_shared_columns[0]
    with st.sidebar:
        selected_column = st.selectbox(
            "Column for bar graph",
            ordered_shared_columns,
            index=ordered_shared_columns.index(default_column),
        )
        chart_mode = st.selectbox("Chart mode", ["Raw rows", "Aggregate by run"], index=0)

    merged_rows: list[pd.DataFrame] = []
    for run in selected_runs:
        run_df = df_by_run[run].copy()
        run_df.insert(0, "run", run)
        run_df.insert(1, "row_id", range(len(run_df)))
        merged_rows.append(run_df)
    merged_df = pd.concat(merged_rows, ignore_index=True)

    st.subheader("Table")
    st.dataframe(merged_df, use_container_width=True, height=500)

    st.subheader("Bar Graph")
    chart_df = merged_df[["run", "row_id", selected_column]].copy()
    chart_df["bar_id"] = chart_df["run"] + " #" + chart_df["row_id"].astype(str)
    numeric_values = pd.to_numeric(chart_df[selected_column], errors="coerce")

    if chart_mode == "Aggregate by run":
        if numeric_values.notna().any():
            with st.sidebar:
                agg_name = st.selectbox("Aggregation", list(NUMERIC_AGGREGATIONS.keys()), index=0)
            agg_fn = NUMERIC_AGGREGATIONS[agg_name]

            agg_rows: list[dict[str, float | str]] = []
            for run in selected_runs:
                run_numeric = pd.to_numeric(df_by_run[run][selected_column], errors="coerce").dropna()
                value = agg_fn(run_numeric) if not run_numeric.empty else float("nan")
                agg_rows.append({"run": run, f"{selected_column}_{agg_name}": value})

            agg_df = pd.DataFrame(agg_rows)
            metric_col = f"{selected_column}_{agg_name}"
            st.bar_chart(agg_df.set_index("run")[metric_col])
            st.dataframe(agg_df, use_container_width=True)
        else:
            agg_name = st.sidebar.selectbox("Aggregation", ["count", "nunique"], index=0)
            agg_rows = []
            for run in selected_runs:
                run_values = df_by_run[run][selected_column]
                if agg_name == "count":
                    value = int(run_values.count())
                else:
                    value = int(run_values.nunique(dropna=True))
                agg_rows.append({"run": run, f"{selected_column}_{agg_name}": value})

            agg_df = pd.DataFrame(agg_rows)
            metric_col = f"{selected_column}_{agg_name}"
            st.bar_chart(agg_df.set_index("run")[metric_col])
            st.dataframe(agg_df, use_container_width=True)
    else:
        if numeric_values.notna().any():
            chart_df["value"] = numeric_values
            numeric_chart_df = chart_df.dropna(subset=["value"]).head(max_rows_for_chart)
            if numeric_chart_df.empty:
                st.info("No plottable values in selected column.")
            else:
                st.bar_chart(numeric_chart_df.set_index("bar_id")["value"])
        else:
            # Keep bar chart behavior for non-numeric columns by encoding categories to integer codes.
            categories = chart_df[selected_column].fillna("NA").astype(str)
            codes, uniques = pd.factorize(categories, sort=True)
            chart_df["value_code"] = codes
            st.bar_chart(chart_df.head(max_rows_for_chart).set_index("bar_id")["value_code"])
            st.caption("Selected column is non-numeric, so bars show encoded category IDs.")
            mapping_df = pd.DataFrame({"category_id": range(len(uniques)), "value": uniques})
            st.dataframe(mapping_df, use_container_width=True)


if __name__ == "__main__":
    main()
