import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 1. Load the speech text file

loader = TextLoader("speech.txt", encoding="utf-8")
raw_documents = loader.load()


# 2. Break the text into smaller chunks

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=256,        # maximum size of each chunk
    chunk_overlap=50,      # small overlap to keep context connected
    length_function=len,
)

chunks = text_splitter.split_documents(raw_documents)
print(f"Text successfully split into {len(chunks)} chunks.")


# 3. Create embeddings (using a free local model)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Build or load the Chroma vector database

DB_PATH = "chroma_db"

if os.path.exists(DB_PATH):
    # If database already exists, load it
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )
    print("Loaded existing Chroma database.")
else:
    # Otherwise create a new database and save it
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print("Created a new Chroma database and saved it.")


# 5. Set up the Local LLM (Ollama with Mistral)

llm = Ollama(
    model="mistral",
    temperature=0.2       
)


# 6. Build a strict prompt (answer ONLY from the text)

prompt_template = """
You are an expert on Dr. B.R. Ambedkar's ideas.
Answer ONLY using the information from the context below.

If the answer is not found in the text, reply:
"I cannot answer based on the provided text."

Context:
{context}

Question: {question}

Answer (quote the text when possible):
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


# 7. Create the Retrieval QA chain

qa_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# 8. Helper function: Ask a question

def ask_question(question: str):
    print(f"\nQuestion: {question}")
    response = qa_pipeline({"query": question})
    print(f"Answer: {response['result']}")
    print("-" * 60)

    # Uncomment to see which chunks were used:
    # for i, source in enumerate(response["source_documents"], start=1):
    #     print(f"[Source {i}]\n{source.page_content}\n")


# 9. Interactive CLI Chat Loop

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AmbedkarGPT – Local RAG Q&A System (Mistral 7B via Ollama)")
    print("="*60)
    print("Ask anything about the speech by Dr. B.R. Ambedkar.")
    print("Type 'quit' / 'exit' anytime to stop.\n")

    while True:
        try:
            query = input("Your question: ").strip()

            if query.lower() in ["quit", "exit", "bye"]:
                print("Goodbye!")
                break

            if query:
                ask_question(query)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
