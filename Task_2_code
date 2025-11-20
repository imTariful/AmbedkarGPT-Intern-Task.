import os
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any
import pandas as pd

from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Evaluation libraries
from ragas.metrics import faithfulness, answer_relevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import nltk
nltk.download('punkt', quiet=True)


# CONFIGURATION

CORPUS_DIR = "corpus"
TEST_DATA_PATH = "test_dataset.json"
DB_ROOT = "chroma_db_eval"
OLLAMA_MODEL = "mistral"

# Compare 3 chunking styles
CHUNK_STRATEGIES = {
    "small":  {"chunk_size": 250, "chunk_overlap": 50},
    "medium": {"chunk_size": 550, "chunk_overlap": 100},
    "large":  {"chunk_size": 900, "chunk_overlap": 150}
}


# LOAD TEST DATA

with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
    test_data = json.load(f)["test_questions"]


# BUILD A RAG PIPELINE FOR ONE CHUNK STRATEGY

def build_rag_pipeline(chunk_size: int, overlap: int, label: str):
    print(f"\nBuilding RAG setup for: {label} chunks...")
    print(f"Chunk Size = {chunk_size}, Overlap = {overlap}")

    # Load all .txt files in the corpus directory
    loader = DirectoryLoader(CORPUS_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    # Split into chunks
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = splitter.split_documents(documents)

    # Embedding model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Build or load vectorstore
    db_path = f"{DB_ROOT}_{label}"
    if os.path.exists(db_path):
        vectorstore = Chroma(
            persist_directory=db_path,
            embedding_function=embedding
        )
    else:
        vectorstore = Chroma.from_documents(
            chunks, embedding, persist_directory=db_path
        )

    # Local LLM
    llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)

    # Prompt used during retrieval QA
    prompt = PromptTemplate(
        template="""Use ONLY the provided context to answer the question.
If the answer is not present, reply "I don't know."

Context:
{context}

Question: {question}
Answer:""",
        input_variables=["context", "question"]
    )

    # Complete RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain



# RETRIEVAL METRICS

def compute_retrieval_metrics(retrieved_docs: List[str], correct_docs: List[str], k=5):
    retrieved = set([os.path.basename(doc) for doc in retrieved_docs[:k]])
    correct = set(correct_docs)

    # Hit: Did we retrieve at least one correct doc?
    hit = len(retrieved & correct) > 0

    # Precision
    precision = len(retrieved & correct) / len(retrieved) if retrieved else 0

    # MRR: rank-based score
    mrr = 0.0
    for rank, doc in enumerate(retrieved_docs[:k], start=1):
        if os.path.basename(doc) in correct:
            mrr = 1 / rank
            break

    return {"hit": hit, "mrr": mrr, "precision_at_k": precision}



# ANSWER METRICS

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_answer_metrics(generated: str, expected: str):
    # Treat empty/"I don't know" as blank
    if not generated.strip() or generated.lower() == "i don't know":
        generated = ""

    # ROUGE-L
    rouge_score = scorer.score(expected, generated)["rougeL"].fmeasure

    # BLEU
    bleu = sentence_bleu(
        [expected.split()],
        generated.split(),
        weights=(0.25, 0.25, 0.25, 0.25)
    )

    return {"rouge_l": rouge_score, "bleu": bleu}



# MAIN EVALUATION LOOP

results = []

for name, params in CHUNK_STRATEGIES.items():
    print("\n" + "=" * 60)
    print(f"Evaluating Chunk Strategy: {name.upper()}")
    print("=" * 60)

    qa_chain = build_rag_pipeline(params["chunk_size"], params["chunk_overlap"], name)

    # Store all results for this chunk strategy
    strategy_data = {"strategy": name, "questions": []}

    hit_scores, mrr_scores, precision_scores = [], [], []

    for entry in tqdm(test_data, desc=f"Testing {name}"):
        question = entry["question"]
        expected_docs = entry["source_documents"]
        expected_answer = entry["ground_truth"]

        # Ask the model
        try:
            output = qa_chain({"query": question})
            predicted_answer = output["result"]
            retrieved_paths = [doc.metadata["source"] for doc in output["source_documents"]]
        except:
            predicted_answer = "Error"
            retrieved_paths = []

        # Retrieval evaluation
        retrieval = compute_retrieval_metrics(retrieved_paths, expected_docs)
        hit_scores.append(retrieval["hit"])
        mrr_scores.append(retrieval["mrr"])
        precision_scores.append(retrieval["precision_at_k"])

        # Answer evaluation
        answer_eval = compute_answer_metrics(predicted_answer, expected_answer)

        # Save per-question details
        strategy_data["questions"].append({
            "id": entry["id"],
            "question": question,
            "generated_answer": predicted_answer,
            "ground_truth": expected_answer,
            "retrieved_docs": [os.path.basename(p) for p in retrieved_paths],
            "correct_docs": expected_docs,
            **retrieval,
            **answer_eval
        })

    # Summaries for this chunk strategy
    strategy_data["summary"] = {
        "hit_rate": np.mean(hit_scores),
        "mrr": np.mean(mrr_scores),
        "precision_at_5": np.mean(precision_scores),
        "avg_rouge_l": np.mean([q["rouge_l"] for q in strategy_data["questions"]]),
        "avg_bleu": np.mean([q["bleu"] for q in strategy_data["questions"]]),
    }

    results.append(strategy_data)


# SAVE RESULTS

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nEvaluation complete! Results saved to test_results.json")


# SUMMARY TABLE

print("\n" + "=" * 80)
print("FINAL COMPARISON (Small vs Medium vs Large Chunks)")
print("=" * 80)

df = pd.DataFrame([
    {
        "Chunk Strategy": r["strategy"],
        "Hit Rate": f"{r['summary']['hit_rate']:.3f}",
        "MRR": f"{r['summary']['mrr']:.3f}",
        "Precision@5": f"{r['summary']['precision_at_5']:.3f}",
        "ROUGE-L": f"{r['summary']['avg_rouge_l']:.3f}",
        "BLEU": f"{r['summary']['avg_bleu']:.3f}",
    }
    for r in results
])

print(df.to_string(index=False))
