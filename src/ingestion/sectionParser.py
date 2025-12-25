# src/ingestion/sectionParser.py
import re
from typing import List, Dict

# Multiple patterns to catch different section formats in tax code
SECTION_PATTERNS = [
    re.compile(r"(§\s*\d+[A-Za-z\-]*\.?\s+[A-Z][^\n§]+)", re.MULTILINE),  # § 164. State taxes
    re.compile(r"(Sec\.\s*\d+[A-Za-z\-]*\.?\s+[A-Z][^\n]+)", re.MULTILINE),  # Sec. 164. State taxes
    re.compile(r"(Section\s+\d+[A-Za-z\-]*\.?\s+[A-Z][^\n]+)", re.MULTILINE)  # Section 164
]

def split_by_section(text: str, page_num: int = None) -> List[Dict]:
    """
    Split text into sections based on tax code section headers.
    Returns list of dicts with header, text, and page metadata.
    """
    # Try each pattern until we find matches
    matches = []
    for pattern in SECTION_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            break
    
    if not matches:
        # No sections found - return entire text as one section
        return [{
            "header": f"Page {page_num}" if page_num else "Unknown Section",
            "text": text.strip(),
            "page": page_num
        }]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        header = match.group(1).strip()
        # Clean up header - remove extra spaces
        header = " ".join(header.split())
        
        body = text[start:end].strip()

        sections.append({
            "header": header,
            "text": body,
            "page": page_num
        })

    return sections


def parse_documents(docs: List) -> List[Dict]:
    """
    Parse documents into sections.
    Handles case where sections span multiple pages.
    
    Args:
        docs: List of Document objects from langchain with page_content and metadata
        
    Returns:
        List of section dicts with header, text, and page
    """
    all_sections = []
    
    for doc in docs:
        page_num = doc.metadata.get("page", 1)
        page_text = doc.page_content
        
        sections = split_by_section(page_text, page_num)
        all_sections.extend(sections)
    
    # Merge sections with same header (spanning multiple pages)
    merged_sections = {}
    for section in all_sections:
        header = section["header"]
        if header in merged_sections:
            # Append text, keep earliest page number
            merged_sections[header]["text"] += "\n" + section["text"]
            merged_sections[header]["page"] = min(
                merged_sections[header]["page"], 
                section["page"]
            )
        else:
            merged_sections[header] = section
    
    return list(merged_sections.values())