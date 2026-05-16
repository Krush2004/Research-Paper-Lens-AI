from pydantic import BaseModel
from typing import List, Dict, Optional


class PaperUploadResponse(BaseModel):
    paper_id: str
    metadata: Dict[str, str]
    sections: List[str]
    equation_count: int
    reference_count: int
    chunks_indexed: int
    message: str


class AnalysisResponse(BaseModel):
    paper_id: str
    summary: Dict[str, str]
    math_explanation: str
    knowledge_gaps: str
    innovation_analysis: str
    reproducibility_analysis: str
    quiz: str
    roadmap: str
    code: str
    diagram: str
    ml_reproducibility: Dict
    ml_difficulty: Dict
    feature_importance: Dict


class ChatRequest(BaseModel):
    paper_id: str
    question: str


class ChatResponse(BaseModel):
    answer: str
    context_chunks: int
