from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableParallel , RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from pathlib import Path
# LangChain - Document class can be in different places depending on LC version
try:
    from langchain.schema import Document        # newer versions
except Exception:
    from langchain.docstore.document import Document  # fallback
import json
import os




# Config you can tweak

os.environ['HUGGINGFACEHUB_API_KEY'] = 'replace by your api'
PAPERS_DIR = Path(r"E:\Programming\Artificial Intelligence\Generative AI\Chatbot\NASA-SpaceX\research_paper")
CHUNK_SIZE = 1000      # characters per chunk (adjust later)
CHUNK_OVERLAP = 200    # characters overlap between chunks
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = Path("faiss_index")


# # step_2_open_one_json.py
# sample_path = PAPERS_DIR / "0001-mice-in-bion-m-1-space-mission-training-and-selection.json"
# print("Exists:", sample_path.exists())

# text = sample_path.read_text(encoding="utf-8")
# # Show first 800 chars so you can inspect
# print(text[:1200])




# step_3_load_function.py
def load_papers_from_json(folder_path):
    folder = Path(folder_path)
    docs = []
    for path in sorted(folder.glob("*.json")):
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as e:
            print(f"Skipping {path.name}: parse error -> {e}")
            continue

        # 1) Prefer a root "text"
        text = None
        if isinstance(data.get("text"), str) and data.get("text").strip():
            text = data["text"].strip()
        # 2) else, fallback to the first chunk's text
        elif "chunks" in data and isinstance(data["chunks"], list) and len(data["chunks"]) > 0:
            first_chunk = data["chunks"][0]
            text = first_chunk.get("text", "").strip()
        else:
            print(f"Skipping {path.name}: no usable text found.")
            continue

        # 3) Build metadata so we always know the source
        metadata = {
            "source_file": path.name,
            "paper_file": data.get("file", "")
        }

        # 4) Create a LangChain Document
        doc = Document(page_content=text, metadata=metadata)
        docs.append(doc)

    print(f"Loaded {len(docs)} document(s) from {folder}")
    return docs

# run it
paper_docs = load_papers_from_json(PAPERS_DIR)

# # Inspect first doc
# if paper_docs:
#     print("First doc metadata:", paper_docs[0].metadata)
#     print("First doc text (first 400 chars):")
#     print(paper_docs[0].page_content[:400])





# step_4_chunking.py
def chunk_documents(paper_docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = splitter.split_documents(paper_docs)   # returns list[Document]
    # Improve metadata: add paper filename and a per-paper chunk id
    new_docs = []
    for doc in split_docs:
        md = dict(doc.metadata or {})
        # if splitter created chunk-level metadata it might not have chunk ids;
        # we'll create a readable chunk id using the source_file.
        source = md.get("source_file", "unknown")
        # create a safe chunk_id using length of current list for that source
        # get current count for this source
        existing_count = sum(1 for d in new_docs if d.metadata.get("source_file") == source)
        md["chunk_id"] = f"{source}_chunk_{existing_count}"
        new_doc = Document(page_content=doc.page_content, metadata=md)
        new_docs.append(new_doc)
    print(f"Chunked into {len(new_docs)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return new_docs

# run it
import pickle

CHUNKED_DOCS_FILE = Path("chunked_docs.pkl")

if CHUNKED_DOCS_FILE.exists():
    print("Loading chunked documents from disk...")
    with open(CHUNKED_DOCS_FILE, "rb") as f:
        chunked_docs = pickle.load(f)
    print(f"Loaded {len(chunked_docs)} chunks from {CHUNKED_DOCS_FILE}")
else:
    print("Chunked file not found. Splitting paper documents into chunks...")
    chunked_docs = chunk_documents(paper_docs, chunk_size=500, chunk_overlap=100)
    # Save to disk for future runs
    with open(CHUNKED_DOCS_FILE, "wb") as f:
        pickle.dump(chunked_docs, f)
    print(f"Saved {len(chunked_docs)} chunks to {CHUNKED_DOCS_FILE}")


# # inspect first 3 chunks
# for i, c in enumerate(chunked_docs[:3]):
#     print("---- chunk", i, "metadata:", c.metadata)
#     print(c.page_content[:300].replace("\n", " "), "\n")
# Load chunked_docs without recomputing
with open(CHUNKED_DOCS_FILE, "rb") as f:
    chunked_docs = pickle.load(f)

print(f"Loaded {len(chunked_docs)} chunks from {CHUNKED_DOCS_FILE}")



# step_5_build_faiss.py
def build_vector_store(docs, persist_dir=PERSIST_DIR, model_name=EMBEDDING_MODEL):
    print("Initializing embeddings with model:", model_name)
    embeddings = HuggingFaceEmbeddings(model_name=model_name)  # may download model
    print("Embedding model ready. Creating FAISS index (embedding documents)...")
    vector_store = FAISS.from_documents(docs, embeddings)
    # Save to disk so you can load later without recomputing
    persist_dir.mkdir(parents=True, exist_ok=True)
    try:
        vector_store.save_local(str(persist_dir))
        print("Saved FAISS index to", persist_dir)
    except Exception as e:
        print("Warning: save_local failed - trying persist()", e)
        try:
            vector_store.persist(str(persist_dir))
            print("Persisted FAISS index to", persist_dir)
        except Exception as e2:
            print("Warning: persisting failed:", e2)
    return vector_store

# run it (this may take time)
if PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir()):
    print("Loading existing FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.load_local(
        str(PERSIST_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )
    print('loaded successfully.')
else:
    print("Building FAISS index for the first time...")
    vs = build_vector_store(chunked_docs)








# step_6_retrieval_test.py
query = "mice behavior after space flight"  # change to your test query
# Option A: simple FAISS method (langchain wrapper)
results = vs.similarity_search(query, k=1)   # returns list[Document]
print("Found", len(results), "chunks.")
for r in results:
    print("-> source:", r.metadata.get("source_file"), "| chunk_id:", r.metadata.get("chunk_id"))
    print(r.page_content[:400].replace("\n", " "), "\n---\n")

# Option B: use retriever API (fixed deprecation warning)
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k":3})
docs = retriever.invoke(query)  # Changed from get_relevant_documents to invoke
print("Retriever returned:", len(docs))






# step_7_load_saved.py - FIXED: Added allow_dangerous_deserialization=True
emb = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)  # you still need embeddings object
vs2 = FAISS.load_local(
    str(PERSIST_DIR), 
    emb, 
    allow_dangerous_deserialization=True  # Added this parameter
)
print("Loaded index. nb vectors:", vs2.index.ntotal if hasattr(vs2.index, "ntotal") else "unknown")



# step_8_format_docs.py
def format_docs(retrieved_docs):
    pieces = []
    for doc in retrieved_docs:
        src = doc.metadata.get("source_file", "unknown")
        pieces.append(f"[{src}] {doc.page_content.strip()}")
    return "\n\n---\n\n".join(pieces)

ctx = format_docs(results)
print(ctx[:2000])
