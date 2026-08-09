"""Streamlit entrypoint. Business logic lives entirely in agent/ and utils/ —
this file is presentation only."""

import streamlit as st

from agent.graph import build_graph
from utils.config import MAX_FILE_MB
from utils.file_loader import load_dataframe, check_file_size, FileLoadError, SUPPORTED_EXTENSIONS
from utils.validation import validate_and_clean

st.set_page_config(page_title="Autonomous Data Analysis Agent", layout="wide")
st.title("Autonomous Data Analysis Agent")
st.caption(
    "Upload a CSV or Excel file. A LangGraph agent will profile it, generate "
    "insights, verify every numeric claim against the actual computed "
    "statistics, choose visualizations, and write a report — autonomously."
)
st.caption(f"Supported: {', '.join(SUPPORTED_EXTENSIONS)} · up to {MAX_FILE_MB}MB")

NODE_LABELS = {
    "profiler_node": "Profiling data...",
    "stats_node": "Computing statistics...",
    "insight_generation_node": "Generating insights...",
    "insight_verification_node": "Verifying insights against the data...",
    "visualization_spec_node": "Selecting visualizations...",
    "chart_rendering_node": "Rendering charts...",
    "report_synthesis_node": "Synthesizing report...",
}

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file:
    try:
        check_file_size(uploaded_file.size, MAX_FILE_MB)
        raw_df = load_dataframe(uploaded_file, uploaded_file.name)
        df, validation_warnings, all_null_columns = validate_and_clean(raw_df)
    except FileLoadError as e:
        st.warning(str(e))
        st.stop()

    # --- Dataset preview + data-quality summary, shown before running the
    # agent so the user knows what they're about to analyze. ---
    with st.expander("Dataset preview", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        null_cells = int(df.isna().sum().sum())
        c3.metric("Missing cells", f"{null_cells:,}")
        st.dataframe(df.head(10), use_container_width=True)
        if validation_warnings:
            st.caption("⚠️ " + " · ".join(validation_warnings))

    graph = build_graph()
    final_state = None

    with st.status("Running analysis agent...", expanded=True) as status:
        try:
            for step in graph.stream(
                {"file_path": uploaded_file.name, "df": df, "retry_count": 0,
                 "errors": list(validation_warnings)}
            ):
                for node_name, node_state in step.items():
                    label = NODE_LABELS.get(node_name, node_name)
                    st.write(label)
                    final_state = {**(final_state or {}), **node_state}
            status.update(label="Analysis complete", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label="Analysis failed", state="error")
            st.error(f"The analysis pipeline hit an unrecoverable error: {exc}")
            st.stop()

    if final_state is None:
        st.error("The pipeline did not return a result.")
        st.stop()

    errors = final_state.get("errors", [])
    if errors:
        with st.expander(f"⚠️ {len(errors)} warning(s) during analysis"):
            for e in errors:
                st.write(f"- {e}")

    evidence = final_state.get("evidence", [])
    if evidence:
        n_passed = sum(1 for e in evidence if e.get("passed"))
        with st.expander(f"🔍 Insight verification: {n_passed}/{len(evidence)} claims verified"):
            for e in evidence:
                icon = "✅" if e.get("passed") else "❌"
                st.write(f"{icon} **{e['claim']}** — claimed `{e.get('claimed_value')}`, "
                         f"actual `{e.get('actual_value')}`")

    st.markdown(final_state.get("final_report", "_No report generated._"))

    chart_paths = final_state.get("chart_paths", [])
    if chart_paths:
        st.subheader("Charts")
        cols = st.columns(2)
        for i, chart_path in enumerate(chart_paths):
            with cols[i % 2]:
                st.image(chart_path)

    st.download_button(
        "Download Report (Markdown)",
        final_state.get("final_report", ""),
        file_name="report.md",
        mime="text/markdown",
    )
else:
    st.info("Upload a file to get started, or try one of the sample datasets in `sample_data/`.")
