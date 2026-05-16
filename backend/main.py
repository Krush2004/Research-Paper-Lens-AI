from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import fitz  # PyMuPDF

from backend.core.parser import ResearchPaperParser
from backend.core.rag import RAGEngine
from backend.core.engine import AIEngine
from backend.core.predictor import MLResearchPredictor
from backend.models.schemas import PaperUploadResponse, AnalysisResponse, ChatRequest, ChatResponse

# ── Initialize FastAPI ──
app = FastAPI(
    title="ResearchLens AI",
    description="Intelligent AI-Powered Research Paper Understanding API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Core Engines ──
print("🚀 Initializing ResearchLens AI engines...")
rag_engine = RAGEngine()
ai_engine = AIEngine()
ml_predictor = MLResearchPredictor()
print("✅ All engines ready!")

# ── In-memory storage for parsed papers (no disk uploads) ──
paper_store: dict = {}


@app.get("/")
async def root():
    return {"message": "ResearchLens AI API is running", "version": "1.0.0"}


@app.post("/upload", response_model=PaperUploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    """Upload and parse a research paper PDF (processed in-memory, not saved to disk)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Read PDF bytes into memory (no disk storage)
        pdf_bytes = await file.read()
        paper_id = str(uuid.uuid4())

        # Parse PDF directly from bytes using PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parser = ResearchPaperParser.__new__(ResearchPaperParser)
        parser.doc = doc
        parser.full_text = ""
        parser.pages = []
        parser.sections = {}
        parser.equations = []
        parser.references = []
        parser.metadata = {}
        parser.file_path = file.filename

        data = parser.parse()

        # Index in Qdrant via langchain-qdrant
        chunks_indexed = rag_engine.index_paper(
            paper_id=paper_id,
            text=data["full_text"],
            metadata=data["metadata"],
        )

        # Store parsed data in memory for analysis
        paper_store[paper_id] = data

        return PaperUploadResponse(
            paper_id=paper_id,
            metadata=data["metadata"],
            sections=list(data["sections"].keys()),
            equation_count=len(data["equations"]),
            reference_count=len(data["references"]),
            chunks_indexed=chunks_indexed,
            message=f"Paper '{file.filename}' processed successfully (in-memory).",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.get("/analyze/{paper_id}", response_model=AnalysisResponse)
async def analyze_paper(paper_id: str):
    """Run full AI + ML analysis on a previously uploaded paper."""
    if paper_id not in paper_store:
        raise HTTPException(status_code=404, detail="Paper not found. Please upload first.")

    data = paper_store[paper_id]
    text = data["full_text"]
    equations = data["equations"]
    sections = data["sections"]
    references = data["references"]

    # Helper for graceful error handling per feature
    def safe_run(func, *args, default="Analysis temporarily unavailable. Please retry."):
        try:
            return func(*args)
        except Exception as e:
            print(f"⚠️ Error in {func.__name__}: {e}")
            return default

    # ── Run all AI features with error handling ──
    summary = safe_run(ai_engine.generate_summary, text, default={"beginner": "Error", "technical": "Error"})
    math = safe_run(ai_engine.explain_math, equations)
    gaps = safe_run(ai_engine.detect_knowledge_gaps, text)
    innovation = safe_run(ai_engine.analyze_innovation, text)
    repro_analysis = safe_run(ai_engine.predict_reproducibility, text)
    quiz = safe_run(ai_engine.generate_quiz, text)
    roadmap = safe_run(ai_engine.generate_roadmap, text)
    code = safe_run(ai_engine.generate_code, text)
    diagram = safe_run(ai_engine.generate_diagram, text)

    # ── Run ML predictions (5-model ensemble) ──
    ml_repro = safe_run(ml_predictor.predict_reproducibility, text, equations, sections, references, default={})
    ml_diff = safe_run(ml_predictor.predict_difficulty, text, equations, sections, references, default={})
    feat_imp = safe_run(ml_predictor.get_feature_importance, default={})

    return AnalysisResponse(
        paper_id=paper_id,
        summary=summary if isinstance(summary, dict) else {"beginner": summary, "technical": summary},
        math_explanation=math,
        knowledge_gaps=gaps,
        innovation_analysis=innovation,
        reproducibility_analysis=repro_analysis,
        quiz=quiz,
        roadmap=roadmap,
        code=code,
        diagram=diagram,
        ml_reproducibility=ml_repro,
        ml_difficulty=ml_diff,
        feature_importance=feat_imp,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_paper(request: ChatRequest):
    """Chat with the paper using RAG (retrieval-augmented generation)."""
    if request.paper_id not in paper_store:
        raise HTTPException(status_code=404, detail="Paper not found. Please upload first.")

    try:
        data = paper_store[request.paper_id]
        title = data["metadata"].get("title", data["metadata"].get("filename", "Unknown Paper"))
        
        # Retrieve relevant context from Qdrant
        retrieved_context = rag_engine.get_context(
            query=request.question,
            paper_id=request.paper_id,
            top_k=5,
        )

        # Inject overarching context (title + first 1500 chars/Abstract) for generic queries
        paper_intro = data["full_text"][:1500]
        context = f"Paper Title: {title}\n\nOverarching Context (Abstract/Intro):\n{paper_intro}\n\nRetrieved Chunks:\n{retrieved_context}"

        # Generate answer using LLM + context
        answer = ai_engine.chat(request.question, context)

        return ChatResponse(answer=answer, context_chunks=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")


# ── Run with uvicorn ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
