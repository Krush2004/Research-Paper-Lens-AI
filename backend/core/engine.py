from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Dict, Any

from backend.config import get_settings

settings = get_settings()


class AIEngine:
    """AI analysis engine using Llama 3.1 via OpenRouter for deep paper understanding."""

    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENROUTER_API_KEY,
            openai_api_base=settings.OPENROUTER_BASE_URL,
            model_name=settings.LLM_MODEL,
            temperature=0.7,
            max_tokens=2048,
        )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content

    # ── Feature 1: Paper Summary ──
    def generate_summary(self, text: str) -> Dict[str, str]:
        """Generate beginner-friendly and technical summaries."""
        system = "You are an expert research paper analyst. Provide clear, structured summaries."
        prompt = f"""Analyze this research paper and provide TWO summaries:

1. **Beginner Summary**: Explain like I'm a college student. Use analogies and simple language.
2. **Technical Summary**: A detailed technical overview for researchers.

Paper text (first 4000 chars):
{text[:4000]}"""
        result = self._call_llm(system, prompt)
        return {"beginner": result, "technical": result}

    # ── Feature 2: Math Equation Explainer ──
    def explain_math(self, equations: List[str]) -> str:
        """Explain detected mathematical equations in plain English."""
        if not equations:
            return "No significant mathematical equations were detected in this paper."
        system = "You are a math tutor who explains complex equations simply."
        eq_text = "\n".join([f"Equation {i+1}: {eq}" for i, eq in enumerate(equations[:10])])
        prompt = f"""Explain each of these equations from a research paper:

{eq_text}

For each equation provide:
- **Variables**: What each symbol represents
- **Intuition**: What is this equation doing in plain English
- **Purpose**: Why is this equation important in the paper
- **Real-world analogy**: A simple analogy to understand it"""
        return self._call_llm(system, prompt)

    # ── Feature 3: Knowledge Gap Detector ──
    def detect_knowledge_gaps(self, text: str) -> str:
        """Identify prerequisite concepts needed to understand the paper."""
        system = "You are an academic advisor helping students identify knowledge gaps."
        prompt = f"""Analyze this research paper and identify ALL prerequisite concepts a reader needs.

Organize your response as:

### 🔴 Critical Prerequisites (Must Know)
- List concepts that are absolutely required

### 🟡 Helpful Background (Should Know)
- List concepts that make understanding easier

### 🟢 Advanced Topics (Nice to Know)
- List advanced concepts for deeper understanding

### 📚 Mini Learning Guide
For each critical prerequisite, provide a 2-3 sentence explanation.

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 4: Innovation Difference Engine ──
    def analyze_innovation(self, text: str) -> str:
        """Compare innovations with previous approaches."""
        system = "You are a research analyst specializing in identifying novel contributions."
        prompt = f"""Analyze this research paper and identify:

### 🆕 Key Innovations
- What is genuinely new in this paper?

### 🔄 Comparison with Prior Work
- How does this differ from previous approaches?

### 📊 Performance Improvements
- What improvements are claimed? Are they significant?

### ⚠️ Limitations
- What are the acknowledged and unacknowledged limitations?

### 💡 Impact Assessment
- How impactful is this work for the field?

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 5: Reproducibility Analysis ──
    def predict_reproducibility(self, text: str) -> str:
        """Analyze how reproducible the paper's results are."""
        system = "You are a reproducibility expert evaluating research papers."
        prompt = f"""Evaluate the reproducibility of this research paper:

### 📊 Reproducibility Score: X/10

### ✅ What's Provided
- List all implementation details, datasets, hyperparameters mentioned

### ❌ What's Missing
- List missing details that would make reproduction difficult

### 💻 Computational Requirements
- Estimate required hardware and time

### 🛠️ Implementation Difficulty
- Rate: Easy / Medium / Hard / Very Hard
- Explain why

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 6: Quiz Generator ──
    def generate_quiz(self, text: str) -> str:
        """Generate quiz questions from the paper content in JSON format."""
        system = """You are a professor creating exam questions from research papers.
You MUST respond with ONLY valid JSON. No markdown, no extra text.
The JSON must have this exact structure:
{
  "mcq": [
    {
      "question": "What is...?",
      "options": {"A": "Option A text", "B": "Option B text", "C": "Option C text", "D": "Option D text"},
      "correct": "B",
      "explanation": "B is correct because..."
    }
  ],
  "conceptual": [
    {"question": "Explain...", "hint": "Think about..."}
  ],
  "coding": [
    {"question": "Implement...", "hint": "Use..."}
  ],
  "viva": [
    {"question": "Why did the authors...?", "hint": "Consider..."}
  ]
}"""
        prompt = f"""Create a comprehensive quiz from this research paper. Return ONLY valid JSON.

Requirements:
- 5 Multiple Choice Questions (MCQ) with 4 options each, one correct answer, and a brief explanation
- 3 Conceptual Questions testing deep understanding
- 2 Coding Questions about implementation
- 3 Viva Questions an examiner might ask

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 7: Implementation Roadmap ──
    def generate_roadmap(self, text: str) -> str:
        """Generate a step-by-step implementation roadmap."""
        system = "You are a senior ML engineer creating implementation guides."
        prompt = f"""Create a detailed implementation roadmap for this paper:

### 🗺️ Implementation Roadmap

For each step provide:
- **Step N**: Title
- **Description**: What to do
- **Libraries**: Required Python packages
- **Difficulty**: Easy/Medium/Hard
- **Estimated Time**: Hours/Days

### 📦 Required Libraries
List all Python packages needed

### 📊 Datasets
List required datasets and where to find them

### ⚡ Quick Start
A minimal viable implementation plan (simplest version)

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 8: Simplified Code Generator ──
    def generate_code(self, text: str) -> str:
        """Generate simplified starter code based on the paper."""
        system = "You are a coding tutor who writes clean, well-commented Python code."
        prompt = f"""Based on this research paper, generate simplified starter code:

Requirements:
- Use PyTorch or TensorFlow
- Include detailed comments explaining each part
- Keep it beginner-friendly
- Include a simple training loop
- Add example usage at the bottom

The code should implement the CORE idea of the paper in the simplest way possible.

Paper text (first 4000 chars):
{text[:4000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 9: Architecture Diagram Generator ──
    def generate_diagram(self, text: str) -> str:
        """Generate a Mermaid.js diagram of the paper's architecture."""
        system = """You are a technical diagram creator. You ONLY output valid Mermaid.js diagram code.
Do NOT include any markdown code fences. Output ONLY the raw Mermaid syntax.
CRITICAL RULES:
- ALWAYS start with: graph TD
- Use ONLY square bracket labels: A["Label Text"]
- NEVER use parentheses () in labels — they cause parse errors
- NEVER use curly braces {} in labels
- Use simple arrows: A --> B or A -->|"label"| B
- Quote all labels with double quotes inside brackets: A["My Label"]
- Keep node IDs simple: A, B, C, D, E etc.
- No special characters in labels (no &, <, >, etc.)"""
        prompt = f"""Create a Mermaid.js flowchart diagram showing the architecture or methodology of this paper.

Rules:
- Start with 'graph TD'
- Use 8-12 nodes maximum
- Every node label MUST be in square brackets with quotes: A["Label"]
- Use simple arrows: A --> B
- For labeled edges use: A -->|"label text"| B
- Do NOT wrap in code fences
- Do NOT use parentheses in any label

Example format:
graph TD
    A["Input Data"] --> B["Processing"]
    B --> C["Model Training"]
    C --> D["Output"]

Paper text (first 3000 chars):
{text[:3000]}"""
        return self._call_llm(system, prompt)

    # ── Feature 10: RAG Chat ──
    def chat(self, question: str, context: str) -> str:
        """Answer questions using RAG context from the paper."""
        system = """You are ResearchLens AI, an expert research paper mentor.
You will be provided with context extracted from a research paper (including the Title, the Abstract/Intro, and specific retrieved text chunks).
Your goal is to answer the user's question based on this context. 
- If the user asks for a general summary or overview, heavily rely on the 'Overarching Context (Abstract/Intro)'.
- If they ask a specific question, rely on the 'Retrieved Chunks'.
- If the answer cannot be deduced from the provided context at all, say so honestly.
Always act as a helpful, educational AI mentor."""
        prompt = f"""Here is the extracted context from the research paper:

{context}

User question: {question}

Provide a clear, helpful, and professional answer based on the context above."""
        return self._call_llm(system, prompt)
