MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT = True
LLM_MAX_LENGTH = 3000

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
SEPARATOR = "\n"

RETRIEVAL_K = 3

UPLOAD_DIR = "data/uploads"
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".docx")
