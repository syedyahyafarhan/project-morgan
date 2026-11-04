import os
from pypdf import PdfReader
from state import AgentState

def pdf_ingestion_node(state:AgentState) -> AgentState:
    """Reads the PDF file path from state, extracts the text and updates the raw_text"""
    file_path = state.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return {"Error": f"File path not found or wrong: {file_path}"}
    try:
        file = PdfReader(file_path)
        extracted_text = ""

        for i, page in enumerate(file.pages):
            text = page.extract_text()
            if text: #if detected text
                extracted_text += f"\n--- Page {i+1} ---\n" + text
        print(f"[INGESTION] Sucessfully extracted {len(extracted_text)} characters from {file_path}")
        return {"raw_text": extracted_text, "error": None} # updating the agentstate rawtext and error elements
    except Exception as e:
        print(f"[Ingestion Error] {str(e)}")
        return {"error": f"Failed to parse pdf: {str(e)}"}



    


