
import io
import json
import pandas as pd
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="UAT Feedback Analyzer", layout="wide")
st.title("UAT Feedback Analyzer (Excel → LLM → Priority)")

REQUIRED_COL = "User Input / Feedback"

template = """

    Act as a Sr Product Manager  
    Analyze the following user feedback of a User Acceptance testing 
    Feedback : "{feedback}"
    Output ONLY valid JSON. No explanations.

    Return the result in exactly this format:
    Category: [ UX, Technical Bug, Feature Request, or Pass Function] 
    Priority : [Immediate Fix,High Priority , No Change or Low Priority] 
    Justification : [1-Sentence Explaination]
    
        """


def feedback_analyser(df_in: pd.DataFrame, feedback_col: str) -> pd.DataFrame:
    df = df_in.copy(deep=True)

    # Prepare output columns
    df["Category"] = None
    df["Priority"] = None
    df["Justification"] = None

    model = OllamaLLM(model="llama3.2",format="json",temperature=0.0,top_p=0.9,top_k=40,seed=42)
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    total = len(df)
    progress = st.progress(0)
    status = st.empty()

    for i, (idx, row) in enumerate(df.iterrows(), start=1):
        fb = row.get(feedback_col, "")
        if pd.isna(fb) or str(fb).strip() == "":
            df.loc[idx, "Justification"] = "Empty feedback."
        else:
            result = chain.invoke({"feedback": str(fb)})
            result_json = json.loads(str(result))

            df.loc[idx, "Category"] = result_json.get("Category")
            df.loc[idx, "Priority"] = result_json.get("Priority")
            df.loc[idx, "Justification"] = result_json.get("Justification")

        progress.progress(i / max(total, 1))
        status.write(f"Processing {i}/{total} rows...")

    status.write("Done ✅")
    return df


uploaded = st.file_uploader("Upload an .xlsx file", type=["xlsx"])
if uploaded is None:
    st.info("Upload an Excel file to begin.")
    st.stop()

try:
    df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Could not read Excel file: {e}")
    st.stop()

st.subheader("1) Preview your file")
st.dataframe(df_raw.head(20), use_container_width=True)

st.subheader("2) Map your columns")
cols = list(df_raw.columns)

# Force user to select which column contains the feedback text
feedback_col = st.selectbox(
    "Select the column that contains User Input / Feedback",
    options=cols,
    index=cols.index("User Input / Feedback") if "User Input / Feedback" in cols else 0,
)

# Optional: let them rename it in output if they want
output_feedback_col_name = st.text_input(
    "Optional: Rename that column in the output (leave blank to keep as-is)",
    value="",
)

# Basic validation
if feedback_col is None or feedback_col.strip() == "":
    st.error("Please select a valid feedback column.")
    st.stop()

run = st.button("Run analysis", type="primary")

if run:
    # Optionally normalize the column name
    df_input = df_raw.copy()
    if output_feedback_col_name.strip():
        df_input = df_input.rename(columns={feedback_col: output_feedback_col_name.strip()})
        feedback_col_to_use = output_feedback_col_name.strip()
    else:
        feedback_col_to_use = feedback_col

    with st.spinner("Calling LLM and enriching your dataset..."):
        df_out = feedback_analyser(df_input, feedback_col_to_use)

    st.subheader("3) Output preview")
    st.dataframe(df_out.head(50), use_container_width=True)

    # Download as Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="Enriched")
    output.seek(0)

    st.download_button(
        label="Download enriched Excel",
        data=output,
        file_name="feedback_enriched.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
