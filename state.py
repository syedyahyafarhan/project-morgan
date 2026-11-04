from typing import Annotated, Sequence, TypedDict, Dict, Optional
class AgentState(TypedDict):
    file_path : str # input, where the pdf is on system.
    raw_text : str # parser output
    extracted_metrics : Dict[str, Any] # strucuted output
    verification_status : str #quality checker, 
    verification_feedback : Optional[str] #critic, notes from the checker put simply
    db_record_id: Optional[int] # row id in the database eventually
    error: Optional[str] #selfexplanatory, capturing crashes if catastrophic failure

