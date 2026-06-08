import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf_files = [
    "papers/Attention Is All You Need.pdf",
    "papers/BERT.pdf",
    "papers/GPT3.pdf",
    "papers/RAG.pdf",
    "papers/SBERT.pdf",
    "papers/LoRA.pdf",
    "papers/LLama2.pdf"
]

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

try:
    chroma_client.delete_collection("research_papers")
except:
    pass

collection = chroma_client.create_collection(
    name="research_papers"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_chunks = []
all_ids = []
all_metadatas = []

chunk_id = 0

for pdf_file in pdf_files:

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    chunks = splitter.split_text(text)

    for chunk in chunks:

        all_chunks.append(chunk)

        all_ids.append(str(chunk_id))

        all_metadatas.append({
            "source": pdf_file.split("/")[-1]
        })

        chunk_id += 1

embeddings = embedding_model.encode(
    all_chunks,
    show_progress_bar=True
)

collection.add(
    documents=all_chunks,
    embeddings=embeddings.tolist(),
    ids=all_ids,
    metadatas=all_metadatas
)

print("Ingestion Complete")
