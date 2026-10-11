# CampusVoice: RAG and LLM chatbot for reporting development challenges in higher learning

Prototype for the paper **"Enhancing Chatbot-Based Reporting of Development Challenges in Higher Learning"** (PRJT302, Group 12), built following the paper's DSRM design.

## Who can do what

| Role | Access |
|---|---|
| **Student** | Chatbot only (`/chat`, no account). Students describe a problem and get an answer grounded in university policy, plus a reference number. To check on a report, they send the reference number to the chatbot. |
| **Staff** | Sign in (`/staff/login`), view and update student reports (`/staff/reports`), and view the model performance dashboard (`/staff/models`). |

There is no registration page, so students cannot create staff accounts. An administrator creates each staff account from the terminal with `flask --app app create-admin`. Passwords are stored as salted hashes.

## Models

The paper's two main approaches, **RAG** and **LLM**, are trained and tested on the uploaded dataset **AI_Chatbots_Students_Attitude_Dataset** (`data/AI_Chatbots_Students_Attitude_Dataset_EN.xlsx`, 131 student responses).

### Prediction task
The dataset is a survey, not complaint text, so the task the models learn is: **predict each student's chatbot adoption level from their attitudes.**

- **Inputs:** 28 Likert items (Q5.1 to Q9.5), degree, gender, field of study and the free-text comment (Q10).
- **Target (Q4, never used as an input):**
  - Low = never or rarely uses AI chatbots (48 students)
  - Medium = sometimes (53)
  - High = often or very often (30)

### RAG (`app/models/attitude_models.py`, `RAGModel`)
1. The training students are stored in a **FAISS vector database**. Each survey item is weighted by how strongly it relates to adoption in the training data.
2. For a new student, the retriever finds the 7 most similar training students.
3. Their profiles and known adoption levels are wrapped into the prompt as grounding context, and the **LLM generates the decision**.
4. Without an LLM configured, RAG uses a similarity-weighted vote over the retrieved students.

### LLM (`LLMZeroShot`)
The language model reads one student's profile and decides with no retrieved context. This is the ungrounded approach that RAG is compared against.

### Evaluation protocol (`app/attitude_eval.py`)
- **Stratified 5-fold cross-validation:** each fold trains on 80% and tests on 20%, so every student is tested exactly once. With only 131 responses, a single 80:20 split would test on just 27 students, and the scores would swing a lot between runs.
- **Metrics:** accuracy, macro-averaged precision, recall and F1-score, plus their standard deviation across folds, confusion matrices and latency.
- **Baseline:** a majority-class baseline is reported as a reference point. A model is only useful if it beats it.

### Chatbot models (complaint handling)
- A deep learning classifier routes each report to Academic, Finance, ICT, Facilities or Hostel: BERT when PyTorch and Transformers are installed, otherwise an MLP neural network.
- RAG over the policy documents in `knowledge_base/` answers the student, with memory, a query rewriter and guardrails.
- The attitude dataset has no complaint text, so the router is trained on `data/sample_complaints.csv` (synthetic), or on the Kaggle complaints CSV from section 3.2 if you save it as `data/university_complaints.csv`.

## Model performance dashboard (single screen)

`/staff/models` shows the RAG and LLM results on one screen with three types of visualisation:

1. **Grouped bar chart:** accuracy, precision, recall and F1 for RAG, LLM and the baseline (hover for the ± across folds).
2. **Radar chart:** the metric profile of each model.
3. **Confusion-matrix heatmaps:** actual versus predicted adoption level for RAG and LLM.

Metric cards at the top show each model's accuracy, precision, recall, F1 and latency. **Re-run evaluation** retrains and retests the models in the background.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/setup_all.py      # builds the vector databases, trains and evaluates all models
python run.py                    # http://localhost:5000
```

1. Students: open http://localhost:5000/chat
2. Staff: create an account from the terminal, then sign in at http://localhost:5000/staff/login

   ```bash
   python -m flask --app app create-admin
   ```

   It asks for the name, email, staff number, department and password (at least 8 characters). Departments: Academic, Finance, ICT, Facilities, Hostel, Student Affairs, Management.
3. Run the tests: `python -m pytest -q`

## Add the LLM (needed for the LLM results and for RAG + LLM generation)

Pick one provider, put it in `.env` (copy `.env.example`), then run `python scripts/evaluate_models.py` or select **Re-run evaluation** on the dashboard.

| Provider | Cost | Setting |
|---|---|---|
| Groq (Llama 3.1 8B) | Free tier | `GROQ_API_KEY=...` from console.groq.com |
| Ollama (Llama 3.2, runs on your laptop) | Free, offline | Install Ollama, run `ollama pull llama3.2`; detected automatically |
| Anthropic Claude | Paid | `ANTHROPIC_API_KEY=...` |
| OpenAI | Paid | `OPENAI_API_KEY=...` |
| Hugging Face Flan-T5 | Free, offline | `pip install -r requirements-full.txt`, `LLM_PROVIDER=hf` |

Each evaluation makes about 260 LLM requests (131 students × 2 models).

## REST API (`GET /api/v1` lists everything)

| Method | Endpoint | Who | Purpose |
|---|---|---|---|
| POST | `/api/v1/chat` | Public | Chatbot. `{"message": "...", "session_id": "..."}`; returns the answer, department, urgency, status, reference and sources. Sending a reference number returns its status. |
| GET | `/api/v1/reports` | Staff | All student reports |
| PATCH | `/api/v1/reports/<reference>` | Staff | `{"status": "Resolved", "note": "..."}` |
| GET | `/api/v1/stats` | Staff | Report statistics |
| GET | `/api/v1/models` | Staff | Models in use |
| POST | `/api/v1/models/rag/predict` | Staff | Predict a student's adoption level from survey answers `{"Q5.1": "Agree", ..., "Q10": "comment"}`; also returns the retrieved students |
| POST | `/api/v1/models/llm/predict` | Staff | Same, with the zero-shot LLM |
| GET | `/api/v1/evaluation` | Staff | Latest accuracy, precision, recall and F1 |
| POST | `/api/v1/evaluation/run` | Staff | Retrain and retest in the background |

Staff endpoints use the session cookie from signing in.

## Reading the results honestly

Without an LLM, retrieval-only RAG currently reaches about **42% accuracy and 34% macro F1**, against a majority-class baseline of 40.5% accuracy and 19% F1.

Predicting how often a student uses chatbots from 131 survey answers is a hard, small-data problem. Even standard machine learning models score around 40% on it. Report the numbers the evaluation actually produces. If they are far below the >90% figures in the literature review, discuss why in the paper: small sample size, an attitude survey instead of complaint text, and three overlapping classes. These are the same limitations listed in your section 2.1.1.

## Project structure

```
app/
  attitude.py           attitude dataset: cleaning, target, profiles, feature vectors
  attitude_eval.py      RAG vs LLM training and testing (stratified k-fold)
  models/
    attitude_models.py  RAG and LLM models for the dataset, majority baseline
    llm.py              LLM client: Anthropic, OpenAI, Groq, Ollama, Hugging Face
    deep_classifier.py  BERT / MLP department classifier for the chatbot
    rag_classifier.py, llm_classifier.py, intent_baseline.py   complaint-routing comparison
  rag/                  FAISS vector store, embeddings, ingestion, memory, rewriter, guardrails, pipeline
  service.py            chatbot flow, status replies, model prediction API
  storage.py            reports, messages and staff accounts (SQLite or Firebase)
  main.py               Flask routes and REST API
  templates/, static/   chat, staff sign-in, reports, model dashboard
data/                   AI_Chatbots_Students_Attitude_Dataset_EN.xlsx, sample_complaints.csv
knowledge_base/         sample university policies (replace with real documents)
scripts/                setup_all, evaluate_models, evaluate_complaints, ingest_kb, train, perception_summary
tests/                  automated tests
```
