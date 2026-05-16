import fitz  # PyMuPDF
import re
from typing import List, Dict


class ResearchPaperParser:
    """Parses research paper PDFs to extract structured content."""

    def __init__(self, file_path: str = ""):
        self.file_path = file_path
        self.doc = None
        self.full_text = ""
        self.pages: List[str] = []
        self.sections: Dict[str, str] = {}
        self.equations: List[str] = []
        self.references: List[str] = []
        self.metadata: Dict[str, str] = {}

    def extract_text(self) -> str:
        """Extract full text from all pages."""
        for page in self.doc:
            page_text = page.get_text()
            self.pages.append(page_text)
            self.full_text += page_text + "\n"
        return self.full_text

    def extract_metadata(self) -> Dict[str, str]:
        """Extract PDF metadata like title and author."""
        meta = self.doc.metadata
        self.metadata = {
            "title": meta.get("title", "").strip(),
            "author": meta.get("author", "Unknown"),
            "subject": meta.get("subject", ""),
            "keywords": meta.get("keywords", ""),
            "page_count": str(len(self.doc)),
            "filename": self.file_path.split("/")[-1] if self.file_path else "Unknown.pdf"
        }
        return self.metadata

    def detect_sections(self) -> Dict[str, str]:
        """Detect common research paper sections using pattern matching."""
        section_patterns = [
            r"(?i)\b(Abstract)\b",
            r"(?i)\b(Introduction)\b",
            r"(?i)\b(Related\s+Work)\b",
            r"(?i)\b(Methodology|Method|Methods|Approach|Proposed\s+Method)\b",
            r"(?i)\b(Experiments?|Experimental\s+Setup)\b",
            r"(?i)\b(Results?|Results?\s+and\s+Discussion)\b",
            r"(?i)\b(Discussion)\b",
            r"(?i)\b(Conclusion|Conclusions?|Summary)\b",
            r"(?i)\b(References|Bibliography)\b",
        ]

        lines = self.full_text.split("\n")
        current_section = "Preamble"
        self.sections[current_section] = ""

        for line in lines:
            matched = False
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    current_section = line.strip()
                    self.sections[current_section] = ""
                    matched = True
                    break
            if not matched:
                self.sections[current_section] = (
                    self.sections.get(current_section, "") + line + "\n"
                )

        return self.sections

    def detect_equations(self) -> List[str]:
        """Detect mathematical equations from the text."""
        patterns = [
            r"\$\$.+?\$\$",
            r"\$.+?\$",
            r"\\begin\{equation\}.+?\\end\{equation\}",
            r"[A-Za-z]+\s*\(.+?\)\s*=\s*.+",
            r"\b\w+\s*=\s*\w+\s*[\+\-\*/]\s*\w+",
            r"[∑∏∫√∂∇∞].+",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, self.full_text, re.DOTALL)
            self.equations.extend(matches)

        # Deduplicate and clean
        seen = set()
        unique = []
        for eq in self.equations:
            eq_clean = eq.strip()
            if eq_clean and eq_clean not in seen and len(eq_clean) > 3:
                seen.add(eq_clean)
                unique.append(eq_clean)
        self.equations = unique[:20]
        return self.equations

    def extract_references(self) -> List[str]:
        """Extract reference entries from the paper."""
        ref_section = ""
        capture = False
        for line in self.full_text.split("\n"):
            if re.match(r"(?i)\b(References|Bibliography)\b", line.strip()):
                capture = True
                continue
            if capture:
                ref_section += line + "\n"

        refs = re.split(r"\n(?=\[?\d+\]?\.?\s)", ref_section)
        self.references = [r.strip() for r in refs if len(r.strip()) > 10][:30]
        return self.references

    def parse(self) -> Dict:
        """Run full parsing pipeline and return structured data."""
        self.extract_text()
        self.extract_metadata()
        self.detect_sections()
        self.detect_equations()
        self.extract_references()

        return {
            "full_text": self.full_text,
            "metadata": self.metadata,
            "sections": self.sections,
            "equations": self.equations,
            "references": self.references,
            "page_count": len(self.doc),
        }
