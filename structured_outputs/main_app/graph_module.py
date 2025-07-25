# graph_module.py

import os
import json
import time
from collections import defaultdict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv(override=True)

llm = ChatOpenAI(model="gpt-4", temperature=0)
parser = JsonOutputParser()

def safe_llm_invoke(prompt, retries=2):
    """
    Utility: Call LLM, parse JSON output, retry with stronger instruction if it fails.
    """
    last_error = None
    for i in range(retries):
        result = llm.invoke(prompt)
        try:
            return parser.parse(result.content)
        except Exception as e:
            last_error = e
            print(f"⚠️ Parse failed, retry {i+1}: {e}")
            prompt = (
                "REPEAT: STRICTLY output the required JSON object only. "
                "Do NOT omit any field. For missing data, fill with null or 'NOT FOUND'.\n"
            ) + prompt
    raise ValueError(f"❌ LLM output could not be parsed as JSON after retries: {last_error}")

# --- Node: Detect Document Type ---
def detect_document_type(state):
    print("\n🟣 [Step 1] Detecting document type...")
    text = state["text"]

    prompt = f"""
Classify the following document as one of these types: "resume", "github_action", or "email".

Text:
{text[:3000]}

Respond ONLY with the label as a JSON string (e.g., "resume"). No explanations, markdown, or extra text.
"""
    doc_type = safe_llm_invoke(prompt)
    if isinstance(doc_type, list):
        doc_type = doc_type[0]
    print(f"✅ Document type: {doc_type}")
    return {**state, "doc_type": doc_type}


# --- Node: Identify Fields Dynamically from Text ---
def identify_fields_agent(state):
    print("\n🟡 [Step 2] Identifying fields...")
    text = state["text"]
    doc_type = state.get("doc_type", "document")

    prompt = f"""
You are a structured field extractor.

Given this {doc_type} document, list ALL field names (in dot notation) that would be relevant for extracting structured data from such a document, including both top-level and nested fields. 
Do NOT try to guess values, just the field names.

Return a flat JSON list of field names as strings.

Text:
{text[:12000]}

Example output:
["name", "email", "education.institution", "skills[]"]

Do not add any explanations or formatting, only return the JSON list.
"""
    fields = safe_llm_invoke(prompt)
    print(f"✅ Found {len(fields)} fields.")
    return {**state, "extraction_fields": fields}


# --- Node: Extract Values in Groups and Chunks ---
def extract_values_agent(state):
    print("\n🟠 [Step 3] Extracting values...")
    text = state["text"]
    fields = state["extraction_fields"]

    # Group fields by prefix for context-relevant extraction
    groups = defaultdict(list)
    for f in fields:
        prefix = f.split('.')[0].split('[')[0]
        groups[prefix].append(f)

    chunks = [text[i:i+3000] for i in range(0, len(text), 3000)]
    combined_results = {}

    for group, group_fields in groups.items():
        field_str = "\n".join(group_fields)
        for chunk in chunks:
            prompt = f"""
Extract the values for the following fields from the provided text chunk.
Return a single-level JSON object with field names as keys.
- If a value is not found in the text, return null for that field.
- For list fields (ending with []), return an array (can be empty).
- Do NOT skip any fields.

Fields to extract:
{field_str}

Text chunk:
{chunk}

Example output:
{{
  "name": "John Doe",
  "email": null,
  "skills[]": ["Python", "Data Analysis"]
}}

Only output the JSON object. No explanation, no markdown, no extra text.
"""
            extracted = safe_llm_invoke(prompt)
            for k, v in extracted.items():
                # Merge results: fill first-found or aggregate unique lists
                if k not in combined_results or not combined_results[k]:
                    combined_results[k] = v
                elif isinstance(v, list) and isinstance(combined_results[k], list):
                    combined_results[k].extend([x for x in v if x not in combined_results[k]])

    print(f"✅ Extracted {len(combined_results)} field values.")
    return {**state, "extracted_values": combined_results}


# --- Node: Fill Final JSON Using Extracted Values and Schema ---
def fill_json_agent(state):
    print("\n🟢 [Step 4] Filling final structured JSON...")
    template = state["output_format"]
    extracted = state["extracted_values"]

    prompt = f"""
You are a JSON assembler.

Given this structured template (as a JSON object) and extracted values (flat JSON key-value pairs, some may be null), fill the template as completely as possible.
- Map values by matching field names (dot notation).
- For missing or null fields, fill in the template with null or "NOT FOUND" (for strings).
- NEVER skip a field from the template, even if not found in extracted values.
- Keep the output in exactly the same structure as the template.

Template:
{json.dumps(template, indent=2)}

Extracted Values:
{json.dumps(extracted, indent=2)}

Only return the filled JSON object, with no extra text or explanation.
"""
    final_json = safe_llm_invoke(prompt)
    print("✅ Final JSON populated.")
    return {"final_json": final_json}


# --- Build the LangGraph ---
builder = StateGraph(dict)
builder.set_entry_point("detect_type")

builder.add_node("detect_type", detect_document_type)
builder.add_node("identify_fields", identify_fields_agent)
builder.add_node("extract_values", extract_values_agent)
builder.add_node("fill_json", fill_json_agent)

builder.add_edge("detect_type", "identify_fields")
builder.add_edge("identify_fields", "extract_values")
builder.add_edge("extract_values", "fill_json")
builder.add_edge("fill_json", END)

graph = builder.compile()
