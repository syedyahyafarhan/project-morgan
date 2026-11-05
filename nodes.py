import os
from pypdf import PdfReader
from state import AgentState, FinancialAdvisorState
from langchain_groq import ChatGroq

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

def extractor_node(state:AgentState) -> AgentState:

    raw_text = state.get("raw_text")
    if not raw_text:
        return {"error": "No raw text found in state to extract metrics from"}
    try:
        llm = ChatGroq(model="gpt-oss-120b", temperature=0) #temp is the variance in replies after same prompt, 0 means same reply every time which is want i think ill want for accuracy
        structured_llm = llm.with_structured_output(FinancialAdvisorState) #structure the llms response to the schema of financialadvisorstate, holy functions batman!

        system_prompt = f"""You are an expert financial auditor. Analyze the following financial document text 
        and extract the key metrics accurately according to the requested schema.
        
        Financial Document Text:
        {raw_text}"""
        result: FinancialAdvisorState = structured_llm.invoke(system_prompt)

        extracted_data = result.model.dump()

        print(f"[Extraction] Successfully extracted metrics: {extracted_data}")
        
        return {"extracted_metrics": extracted_data, "error": None}
    except Exception as e:
        print(f"[Extraction Error] {str(e)}")
        return {"error": f"Failed during LLM extraction: {str(e)}"}




    


