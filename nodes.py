import os
from pypdf import PdfReader
from state import AgentState, FinancialAdvisorState
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
# llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

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
                chunk_size=15000,
                chunk_overlap=500)
            
        
        chunks = text_splitter.create_documents([raw_text])
        
        
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
        print(f"[EXTRACTION] Successfully extracted metrics across chunks: {datajoined}")
        return {"extracted_metrics": datajoined, "error": None}
    except Exception as e:
        print(f"[Extraction Error] {str(e)}")
        return {"error": f"Failed during LLM extraction: {str(e)}"}
def verifier_node(state:AgentState) -> AgentState:
    metrics = state.get("extracted_metrics", {})
    if not metrics:
        return {
            "verification_status": "FAILED",
            "verification_feedback": "Extraction result was completely empty. No metrics found."
        }
    try:
        
        
        system_prompt = f"""
        You are a strict financial data validator and quality assurance auditor. 
        Review the following extracted financial metrics from a corporate filing:
        
        {metrics}
        
        Evaluate the data based on these rules:
        1. Are essential fields like company_name, fiscal_year, and at least one financial metric (total_revenue or net_income) present?
        2. Are there any obvious logical contradictions (e.g., net_income greater than total_revenue)?
        
        Respond ONLY with a JSON object containing two keys:
        - "status": "PASSED" or "FAILED"
        - "feedback": A short, clear explanation of your findings and any data quality concerns.
        """
        class ValidationResult(BaseModel):
            status: str = Field(description="Must be 'PASSED' or 'FAILED'")
            feedback: str = Field(description="Detailed explanation of the validation result")
        validator_llm = llm.with_structured_output(ValidationResult)
        result: ValidationResult = validator_llm.invoke(system_prompt)

        print(f"[VERIFICATION] Status: {result.status} || Feedback: {result.feedback}")

        return{
            "verification_feedback" : result.feedback ,
            "verification_status" : result.status
        }
    except Exception as e:
        print(f"[Verification Error] {str(e)}")
        return {
            "verification_status": "FAILED",
            "verification_feedback": f"Verification node crashed due to error: {str(e)}"
        }                        
        








    


