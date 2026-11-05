import os
from pypdf import PdfReader
from state import AgentState, FinancialAdvisorState
from langchain_groq import ChatGroq
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

def pdf_ingestion_node(state:AgentState) -> AgentState:
    """Reads the PDF file path from state, extracts the text and updates the raw_text"""
    file_path = Path(state.get("file_path"))
    if not file_path or not os.path.exists(file_path):
        return {"Error": f"File path not found or wrong: {file_path}"}
    try:
        reader = PdfReader(file_path)
        extracted_text = "\n".join([page.extract_text() for page in reader.pages])
        print(f"[INGESTION] Sucessfully extracted {len(extracted_text)} characters from {file_path}")
        return {"raw_text": extracted_text, "error": None} # updating the agentstate rawtext and error elements
    except Exception as e:
        print(f"[Ingestion Error] {str(e)}")
        return {"error": f"Failed to parse pdf: {str(e)}"}

def extractor_node(state:AgentState) -> AgentState:
    """
    Takes raw_text from state, prompts an LLM with a Pydantic schema,
    and extracts structured financial metrics.
    """
    raw_text = state.get("raw_text")
    if not raw_text:
        return {"error": "No raw text found in state to extract metrics from"}
    try:
        text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=14000,
                chunk_overlap=500)
            
        
        chunks = text_splitter.create_documents([raw_text])
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
        structured_llm = llm.with_structured_output(FinancialAdvisorState)

        datajoined = {}

        for i,chunk in enumerate(chunks):
            system_prompt = f"""You are an expert financial auditor. Analyze this section (Chunk {i+1} of {len(chunks)}) 
            of the financial document and extract any key metrics matching the requested schema.If data is missing in this chunk, leave fields null.You must always invoke the tool, even if all values are null. Do not output markdown text or explanations
            
            Document Chunk:
            {chunk.page_content}"""

            result: FinancialAdvisorState = structured_llm.invoke(system_prompt)
            partial_data = result.model_dump()

            for key,value in partial_data.items():
                if value is not None and value != "" and value != 0:
                    datajoined[key] = value
        print(f"[Extraction] Successfully extracted metrics across chunks: {datajoined}")
        return {"extracted_metrics": datajoined, "error": None}
    except Exception as e:
        print(f"[Extraction Error] {str(e)}")
        return {"error": f"Failed during LLM extraction: {str(e)}"}

        








    


