import json
from typing import List, Dict, Generator, Any


class Document:
    """Simple document class to replace LangChain's Document"""

    def __init__(self, page_content: str, metadata: Dict[str, Any] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class DataLoader:
    def __init__(self, json_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        self.json_path = json_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        if not text:
            return []

        chunks = []
        paragraphs = text.split("\n\n")
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > self.chunk_size:
                # Add current chunk if it's not empty
                if current_chunk:
                    chunks.append(current_chunk)

                # Start new chunk with overlap from previous chunk
                if current_chunk and self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-min(len(words), self.chunk_overlap):]
                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add final chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def load_documents_stream(self, batch_size: int = 10) -> Generator[List[Document], None, None]:
        """Stream documents in batches to reduce memory usage"""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {str(e)}")

        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            docs = []

            for item in batch:
                text = item.get("text", "")
                chunks = self._split_text(text)

                for chunk in chunks:
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "id": item.get("id", ""),
                            "section": item.get("section", ""),
                            "title": item.get("title", ""),
                        }
                    )
                    docs.append(doc)

            yield docs

    def load_documents(self) -> List[Document]:
        """Load all documents at once (use only for small datasets)"""
        all_docs = []
        for batch in self.load_documents_stream():
            all_docs.extend(batch)
        return all_docs