import gradio as gr
import os
import json
from graph_module import graph
from dotenv import load_dotenv

load_dotenv(override=True)

from PyPDF2 import PdfReader

def extract_text_from_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    with open(file_path, "rb") as f:
        if ext == ".pdf":
            reader = PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif ext in [".txt", ".md"]:
            return f.read().decode("utf-8")
        else:
            raise ValueError(f"Unsupported file type: {ext}")

def run_parser(document, schema):
    progress = ""
    try:
        # Step 1: Detect document type
        progress += "🟣 [Step 1] Detecting document type...\n"
        yield progress

        document_text = extract_text_from_document(document)

        # Step 2: Load schema (accept both .json and .txt files)
        progress += "🟡 [Step 2] Identifying fields...\n"
        yield progress

        with open(schema, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                schema_data = json.loads(content)
            except Exception as e:
                progress += f"\n❌ Error: Schema file is not valid JSON: {e}"
                yield progress
                return

        # Step 3: Extract values
        progress += "🟠 [Step 3] Extracting values...\n"
        yield progress

        # Step 4: Fill final JSON
        progress += "🟢 [Step 4] Filling final JSON...\n"
        yield progress

        # Run the LangGraph pipeline
        result = graph.invoke({
            "text": document_text,
            "output_format": schema_data
        })

        progress += "✅ Done! Here is your structured JSON output:\n"
        progress += json.dumps(result["final_json"], indent=2)
        yield progress

    except Exception as e:
        progress += f"\n❌ Error: {str(e)}"
        yield progress

# --- Gradio UI setup ---
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Document to Structured JSON Extractor")

    with gr.Row():
        doc_input = gr.File(label="Upload Document (.pdf, .txt, .md)", file_types=[".pdf", ".txt", ".md"])
        schema_input = gr.File(label="Upload Schema (.json, .txt)", file_types=[".json", ".txt"])

    output = gr.Textbox(label="Output / Progress", lines=30, interactive=False)

    btn = gr.Button("Run Parser")

    btn.click(fn=run_parser, inputs=[doc_input, schema_input], outputs=output)

demo.launch(share = True)
