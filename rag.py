import os

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

from settings import (ALLOWED_EXTENSIONS, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME,
                      MAX_FILE_SIZE_MB, RETRIEVAL_K, SEPARATOR)

text_splitter = CharacterTextSplitter(
    separator=SEPARATOR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

documents_store = {}
vectordbs = {}
_embedding = None


class DocumentError(Exception):
    pass


def get_embedding():
    global _embedding
    if _embedding is None:
        _embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embedding


def validate_file(file_path):
    file_name = os.path.basename(file_path)
    extension = os.path.splitext(file_name)[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise DocumentError(f"'{file_name}' is not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    if not os.path.exists(file_path):
        raise DocumentError(f"'{file_name}' was not found.")
    size = os.path.getsize(file_path)
    if size == 0:
        raise DocumentError(f"'{file_name}' is empty.")
    if size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise DocumentError(f"'{file_name}' is larger than {MAX_FILE_SIZE_MB} MB.")
    return file_name, extension


def load_document(file_path):
    file_name, extension = validate_file(file_path)
    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        elif extension == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            loader = Docx2txtLoader(file_path)
        pages = loader.load()
    except Exception as error:
        raise DocumentError(
            f"Could not read '{file_name}'. It may be corrupted or password-protected."
        ) from error
    if not any(p.page_content.strip() for p in pages):
        raise DocumentError(
            f"No text found in '{file_name}'. It may be a scanned document (images only)."
        )
    return file_name, pages


def add_document(file_path):
    name, pages = load_document(file_path)
    if name in vectordbs:
        raise DocumentError(f"'{name}' is already uploaded.")

    chunks = text_splitter.split_documents(pages)
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["doc_name"] = name
        chunk.metadata["chunk_id"] = chunk_id

    try:
        vectordbs[name] = FAISS.from_documents(chunks, get_embedding())
    except Exception as error:
        raise DocumentError(f"Could not index '{name}'. Please try again.") from error
    documents_store[name] = pages
    return name


def list_documents():
    return list(vectordbs.keys())


def remove_document(name):
    vectordbs.pop(name, None)
    documents_store.pop(name, None)


def clear_documents():
    vectordbs.clear()
    documents_store.clear()


def get_full_text(name):
    return "\n".join(page.page_content for page in documents_store[name])


def get_doc_info(name):
    pages = documents_store[name]
    return {
        "pages": len(pages),
        "words": sum(len(p.page_content.split()) for p in pages),
        "type": os.path.splitext(name)[1].lower().strip(".").upper(),
    }


def retrieve(query, doc_name=None, k=RETRIEVAL_K):
    if doc_name and doc_name not in vectordbs:
        raise DocumentError(f"'{doc_name}' is not loaded.")
    names = [doc_name] if doc_name else list_documents()
    if not names:
        raise DocumentError("No documents uploaded yet.")
    results = []
    for name in names:
        results += vectordbs[name].similarity_search(query, k=k)
    return results


def doc_label(doc):
    page = f"page {doc.metadata['page'] + 1}, " if "page" in doc.metadata else ""
    return f"{doc.metadata['doc_name']} - {page}chunk {doc.metadata['chunk_id']}"


def make_references(docs, snippet_len=150):
    references, seen = [], set()
    for doc in docs:
        key = (doc.metadata["doc_name"], doc.metadata["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        # the loader counts pages from 0, so the page is +1
        references.append({
            "document": doc.metadata["doc_name"],
            "page": doc.metadata["page"] + 1 if "page" in doc.metadata else None,
            "chunk": doc.metadata["chunk_id"],
            "snippet": doc.page_content[:snippet_len].replace("\n", " ") + "...",
        })
    return references


def format_reference(reference):
    parts = [reference["document"]]
    if reference["page"] is not None:
        parts.append(f"Page {reference['page']}")
    parts.append(f"Chunk {reference['chunk']}")
    return " — ".join(parts)
