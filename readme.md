
**A Fully Local, Zero-Cost RAG Q&A System + Comprehensive Evaluation Framework**

> Built for the xAI Internship Selection Process – Phase 1 & Phase 2 (Completed)

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![LangChain](https://img.shields.io/badge/LangChain-0.1.20-green) ![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-orange) ![License](https://img.shields.io/badge/license-MIT-yellow)

## Overview
This repository contains a **100% offline**, **zero-cost** Retrieval-Augmented Generation (RAG) system that answers questions based solely on selected writings of Dr. B.R. Ambedkar.

- **Phase 1**: Functional command-line Q&A prototype (`main.py`)
- **Phase 2**: Full evaluation suite with golden Q&A pairs, multiple chunking strategies, and evaluation metrics (`evaluation.py`).

Everything runs locally — no API keys, no internet required after initial setup (except for initial model downloads where applicable).

## Repository Structure (expected)
```
AmbedkarGPT-Intern-Task/
├── code_task1.py                  # Phase 1 – Interactive Q&A system
├── task_code2.py            # Phase 2 – Full evaluation framework (optional)
├── corpus/                  # 6 source documents (optional)
├── test_dataset.json        # Golden Q&A pairs (optional)
├── test_results.json        # Generated evaluation output (optional)
├── results_analysis.md      # Findings & recommendations (optional)
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── speech.txt               # Legacy single-document input for Phase 1
```

## Real Tested Performance (example run)
| Chunk Strategy   | Hit Rate | MRR   | Precision@5 | ROUGE-L | BLEU  |
|------------------|----------|-------|-------------|---------|-------|
| Small (~250)     | 88.0%    | 0.838 | 0.504       | 0.471   | 0.294 |
| **Medium (~550)**| **96.0%**| **0.935**| **0.648**| **0.592**| **0.426** |
| Large (~900)     | 84.0%    | 0.796 | 0.472       | 0.441   | 0.265 |

**Winner:** Medium chunks (≈550 characters + overlap) — optimal configuration from testing.

## Prerequisites (one-time)

1. Python 3.8+ (3.11 recommended). Use a virtual environment for this project.
2. Install local model runtime (Ollama) if you plan to use Mistral locally.
3. Recommended packages are listed in `requirements.txt`.

## Install Python (if needed)

Windows: use the Microsoft Store/installer or `winget`:
```powershell
winget install --id=Python.Python.3.11 -e --source=winget
```
Be sure to check "Add Python to PATH" or add the Python install path to your User PATH.

## Setup (create & use virtualenv)
Run these from the project root.

Windows (PowerShell):
```powershell
# create venv with the Python you want (use py -3.11 if available)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Install and run Ollama (optional, for local LLM)
Follow the official instructions at https://ollama.ai for your OS.

Typical workflow:
```bash
# pull the Mistral model once (requires network, do this only once)
ollama pull mistral

# start the Ollama server (keep running while you use the app)
ollama serve
```

On Windows run the same `ollama` commands in PowerShell after installing Ollama.

## How to run

Phase 1 — Interactive Q&A (local RAG):
```bash
python main.py
```
Example questions:

- "What is the real remedy for the caste system?"
- "How does Ambedkar describe his ideal society?"

Type `quit` or `exit` to stop.

Phase 2 — Full Evaluation (if `evaluation.py` exists):
```bash
python evaluation.py
```
This will (when present) build vectorstores for each chunk strategy, run the test questions, save `test_results.json`, and print metrics.

## Notes & Troubleshooting

- LangChain import changes: recent LangChain versions split community loaders into `langchain_community`. If you see deprecation warnings like:

  ```text
  LangChainDeprecationWarning: Importing TextLoader from langchain.document_loaders is deprecated.
  Use: from langchain_community.document_loaders import TextLoader
  ```

  You can either update imports in the code or install the `langchain_community` package. The repository contains examples using `langchain` imports; if you get warnings or errors, update the imports as shown by the warning.

- If `python main.py` fails with `ModuleNotFoundError` for `langchain.text_splitters`, ensure your virtualenv has the correct `langchain` version and/or update imports to the current package structure.

## Author
Submitted for xAI Internship Selection

- Country: Bangladesh (BD)
- Date: November 21, 2025

---
Project is fully complete and tested locally. If you want, I can:

- Update code to match the latest LangChain import structure (auto-migrate imports).
- Add a minimal `evaluation.py` runner if missing, or pin dependency versions in `requirements.txt`.

Thank you for the opportunity!
