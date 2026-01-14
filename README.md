# UAT Feedback Analyzer (Ollama + LangChain)

This script reads an Excel file containing UAT feedback, calls a local Ollama model (`llama3.2`) to classify each feedback item, and writes the results back into a pandas DataFrame with three new columns:

- `Category` (UX | Technical Bug | Feature Request | Latency)
- `Priority` (P0 | P1 | P2)
- `Justification` (1 sentence)

## Prerequisites

### 1) Python
- Recommended: Python 3.10+

### 2) Ollama (local model runtime)
Install and run Ollama, then pull the model:

```bash
ollama pull llama3.2
```

Make sure Ollama is running (usually it runs as a background service after install). To verify:

```bash
ollama list
```

## Input file requirements

Your Excel file must contain a column named exactly:

- `User Input / Feedback`

The script currently loads:

- `Warehouse_Inventory_UAT_Feedback_40plus.xlsx` from the **current working directory**.

## Setup

Create and activate a virtual environment (recommended):

```bash
python -m venv nvenv
source nvenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Place `Warehouse_Inventory_UAT_Feedback_40plus.xlsx` in the same folder as your script (or run the script from the folder containing the file), then run:

```bash
python your_script_name.py
```

You should see a preview printed to the console:

- `result1.head(5)` (enriched output)
- `df.head(5)` (original input DataFrame, unchanged if you used `.copy(deep=True)`)

## Notes / Troubleshooting

### JSON parsing errors
If the model returns extra text around the JSON, `json.loads()` can fail. Using Ollama JSON mode (`format="json"`) usually helps, but if you still see issues, add a defensive JSON extraction step before `json.loads()`.

### Deployment
This script relies on **local Ollama**. If you deploy to Streamlit Community Cloud, you typically cannot run Ollama there; you’d need a hosted model/API or deploy on your own server that runs Ollama.

## License
Add your preferred license (MIT/Apache-2.0/etc.).
