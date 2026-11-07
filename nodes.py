import os
from pypdf import PdfReader
from state import AgentState, FinancialAdvisorState
from langchain_groq import ChatGroq
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
import json
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter

#llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
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
def extractor_node(state: AgentState) -> AgentState:
    """
    Takes raw_text from state, prompts an LLM with Pydantic schema instructions,
    and extracts structured financial metrics.
    """
    raw_text = state.get("raw_text")
    if not raw_text:
        return {"error": "No raw text found in state to extract metrics from"}
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=12000,
            chunk_overlap=500
        )
        chunks = text_splitter.create_documents([raw_text])
        datajoined = {}

        schema_json = FinancialAdvisorState.model_json_schema()

        for i, chunk in enumerate(chunks):
            system_prompt = f"""You are an expert financial auditor. Analyze this section (Chunk {i+1} of {len(chunks)}) 
            of the financial document and extract key metrics matching the JSON schema below. 
            If data is missing in this chunk, omit the fields or set them to null.
            You must output ONLY a valid JSON object matching the schema. Do not include markdown code blocks (like ```json) or any explanations.

            JSON Schema:
            {schema_json}

            Document Chunk:
            {chunk.page_content}"""

            response = llm.invoke(system_prompt)
            content = response.content.strip()

            # Clean markdown code blocks if the ai does it even if i told it not to..
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:].strip()
                content = content.strip()

            try:
                result = FinancialAdvisorState.model_validate_json(content)
                partial_data = result.model_dump()

                for key, value in partial_data.items():
                    if value is not None and value != "" and value != 0:
                        datajoined[key] = value
            except Exception:
                # Skip chunks that fail JSON parsing or return bad data so we can keep on going
                continue

        print(f"[EXTRACTION] Successfully extracted metrics across chunks: {datajoined}")
        return {"extracted_metrics": datajoined, "error": None}
    except Exception as e:
        print(f"[Extraction Error] {str(e)}")
        return {"error": f"Failed during LLM extraction: {str(e)}"}
def verifier_node(state: AgentState) -> AgentState:
    metrics = state.get("extracted_metrics", {})
    if not metrics:
        return {
            "verification_status": "FAILED",
            "verification_feedback": "Extraction result was completely empty. No metrics found."
        }
    try:
        class ValidationResult(BaseModel):
            status: str = Field(description="Must be 'PASSED' or 'FAILED'")
            feedback: str = Field(description="Detailed explanation of the validation result")

        system_prompt = f"""
        You are a strict financial data validator and quality assurance auditor. 
        Review the following extracted financial metrics from a corporate filing:
        
        {metrics}
        
        Evaluate the data based on these rules:
        1. Are essential fields like company_name, fiscal_year, and at least one financial metric (total_revenue or net_income) present?
        2. Are there any obvious logical contradictions (e.g., net_income greater than total_revenue)?
        
        You MUST output ONLY a valid JSON object with exact keys: "status" ("PASSED" or "FAILED") and "feedback" (string explanation). Do not include markdown code blocks or additional text.
        """
        
        response = llm.invoke(system_prompt)
        content = response.content.strip()

        # Clean markdown code blocks if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()
            content = content.strip()

        validation_result = ValidationResult.model_validate_json(content)

        print(f"[VERIFICATION] Status: {validation_result.status} || Feedback: {validation_result.feedback}")

        return {
            "verification_feedback": validation_result.feedback,
            "verification_status": validation_result.status
        }
    except Exception as e:
        print(f"[Verification Error] {str(e)}")
        return {
            "verification_status": "FAILED",
            "verification_feedback": f"Verification node crashed due to error: {str(e)}"
        }







    


