from typing import Annotated, Sequence, TypedDict, Dict, Optional, Any
from pydantic import BaseModel, Field
class AgentState(TypedDict):
    file_path : str # input, where the pdf is on system.
    raw_text : str # parser output
    extracted_metrics : Dict[str, Any] # strucuted output
    verification_status : str #quality checker, 
    verification_feedback : Optional[str] #critic, notes from the checker put simply
    db_record_id: Optional[int] # row id in the database eventually
    error: Optional[str] #selfexplanatory, capturing crashes if catastrophic failure

class FinancialAdvisorState(BaseModel):
    company_name: Optional[str] = Field(description="Name of company reporting to the financials")
    fiscal_year: Optional[str] = Field(description="The fiscal year or period of the report (e.g., 2025, Q3 2026).")
    total_revenue: Optional[float] = Field(description="Total revenue or sales in numeric format.")
    operating_expenses: Optional[float] = Field(description="Total operating expenses if available.")
    net_income: Optional[float] = Field(description="Net income or net profit/loss in numeric format.")
    currency: Optional[str] = Field(description="Currency of the financial values (e.g., USD, EUR, INR).")
#Elements are optional to avoid AI hallucinations to fill them in otherwise. 