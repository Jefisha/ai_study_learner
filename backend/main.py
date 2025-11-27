# main.py
# ============================================
# AI Student Learner API (FastAPI + LangGraph)
# ============================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import google.generativeai as genai
from langgraph.graph import StateGraph
import chromadb
from chromadb.utils import embedding_functions
import json
import re
import uuid
import time

# -----------------------
# CONFIG
# -----------------------
GEMINI_API_KEY = "AIzaSyDcXLtrGzoNPNWKYMQ_13r1u1NSaHOsiZA"
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# ChromaDB (local persistent store)
client = chromadb.Client(chromadb.config.Settings(persist_directory="study_store"))
embedder = embedding_functions.DefaultEmbeddingFunction()
quiz_collection = client.get_or_create_collection(name="quiz_store", embedding_function=embedder)
eval_collection = client.get_or_create_collection(name="eval_store", embedding_function=embedder)

# In-memory stores (simple demo)
history_store: List[Dict[str, Any]] = []
total_rewards: int = 0

# -----------------------
# LangGraph (kept for compatibility)
# -----------------------
class StudyState(Dict):
    query: str
    level: str
    answer: str
    quiz: List[dict]
    user_answers: List[str]
    results: List[dict]

def answer_query(state: StudyState):
    prompt = f"Answer clearly:\n{state['query']}"
    response = gemini.generate_content(prompt)
    state["answer"] = response.text
    return state

def generate_quiz_node(state: StudyState):
    # This node isn't used directly in /quiz below (we generate directly there),
    # but we keep the node for compatibility with your previous workflow.
    prompt = f"Generate a {state['level']} level quiz (5 questions) based on:\n{state['query']}\nOutput only JSON: [{{'question':'...','answer':'...'}}, ...]"
    response = gemini.generate_content(prompt)
    raw = response.text.strip()
    m = re.search(r'\[.*\]', raw, re.DOTALL)
    if m:
        try:
            state["quiz"] = json.loads(m.group(0))
        except Exception:
            state["quiz"] = []
    else:
        state["quiz"] = []
    return state

graph = StateGraph(StudyState)
graph.add_node("answer_query", answer_query)
graph.add_node("generate_quiz", generate_quiz_node)
graph.set_entry_point("answer_query")
graph.add_edge("answer_query", "generate_quiz")
workflow = graph.compile()

# -----------------------
# FastAPI setup
# -----------------------
app = FastAPI(title="AI Student Learner API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# -----------------------
# Models
# -----------------------
class AskInput(BaseModel):
    question: str

class QuizRequest(BaseModel):
    topic: str
    level: str
    num_questions: int = 5

class SubmitQuizRequest(BaseModel):
    quiz_id: str
    user_answers: List[str]

# -----------------------
# Helpers
# -----------------------

def safe_extract_json_list(text: str):
    """Try to extract a JSON array from LLM output; return Python list or empty list."""
    if not text:
        return []
    # Try to find the first JSON array-looking substring
    m = re.search(r'(\[.*\])', text, re.DOTALL)
    if not m:
        # fallback: try to treat entire response as JSON
        try:
            return json.loads(text)
        except Exception:
            return []
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        # attempt lightweight corrections (replace single quotes -> double quotes)
        try:
            fixed = block.replace("'", '"')
            return json.loads(fixed)
        except Exception:
            return []

def generate_quiz_via_gemini(topic: str, level: str, n: int):
    """Ask Gemini to create n Q/A pairs in JSON. Returns list of {question, answer}."""
    prompt = (
        f"Create a quiz of {n} {level} level questions about '{topic}'. "
        "Return output as pure JSON array of objects with keys 'question' and 'answer'. "
        "Example: [{\"question\":\"...\",\"answer\":\"...\"}, ...]"
    )
    resp = gemini.generate_content(prompt)
    qlist = safe_extract_json_list(resp.text)
    # If LLM returned fewer than requested, try to fill trivially
    if len(qlist) < n:
        # Create placeholder incremental questions if needed (fallback)
        while len(qlist) < n:
            qlist.append({"question": f"{topic} question {len(qlist)+1}", "answer": "Answer not available"})
    return qlist[:n]

# -----------------------
# Routes
# -----------------------
@app.get("/")
def home():
    return {"message": "AI Student Learner FastAPI Running! Visit /docs to test."}

# Ask (doubts) - saves history, awards small points (optional)
@app.post("/ask")
async def ask_api(data: AskInput):
    global total_rewards
    # Use Gemini to answer
    response = gemini.generate_content(data.question)
    answer = response.text

    # Save in history (simple plain-text record)
    history_store.append({"id": str(uuid.uuid4()), "type": "ask", "question": data.question, "answer": answer, "time": time.time()})

    # Reward: small points for asking (optional) — here set to 0 for clarity (we'll award only for correct quiz answers)
    # total_rewards += 0
    return {"answer": answer}

# Quiz generation: returns quiz_id, quiz list (questions only) and answers are stored server-side
@app.post("/quiz")
def create_quiz(req: QuizRequest):
    # generate
    topic = req.topic
    level = req.level
    n = max(1, int(req.num_questions))
    qlist = generate_quiz_via_gemini(topic, level, n)

    # create an id and store (we will store the full quiz JSON as a document)
    quiz_id = f"quiz_{uuid.uuid4().hex}"
    try:
        quiz_collection.add(ids=[quiz_id], documents=[json.dumps({"topic": topic, "level": level, "quiz": qlist})])
    except Exception:
        # if ID collision or store error, ignore for now
        pass

    # Also append to history (generation event)
    history_store.append({"id": quiz_id, "type": "quiz_generated", "topic": topic, "level": level, "time": time.time()})

    # Return questions only (answers stay on server until submit)
    questions_only = [q.get("question", "") for q in qlist]
    return {"quiz_id": quiz_id, "quiz": questions_only, "num_questions": len(questions_only)}

# Submit quiz answers for evaluation: returns score, correct answers, explanations (per question)
@app.post("/submit-quiz")
def submit_quiz(data: SubmitQuizRequest):
    global total_rewards

    # fetch quiz doc from chroma (simple search by id)
    try:
        docs = quiz_collection.get(ids=[data.quiz_id])
        if docs and len(docs["documents"])>0:
            stored = json.loads(docs["documents"][0])
            stored_quiz = stored.get("quiz", [])
            topic = stored.get("topic", "unknown")
        else:
            return {"error": "Quiz not found"}
    except Exception:
        return {"error": "Quiz retrieval error"}

    # Evaluate: compare answers (very simple exact or case-insensitive matching)
    correct_answers = [q.get("answer","") for q in stored_quiz]
    user_answers = data.user_answers
    results = []
    correct_count = 0
    explanations = []

    for idx, correct in enumerate(correct_answers):
        ua = user_answers[idx] if idx < len(user_answers) else ""
        # simple normalize
        def norm(s): return re.sub(r'\s+',' ', str(s).strip().lower())
        is_correct = norm(ua) == norm(correct)
        if not is_correct:
            # ask Gemini for short explanation for the user's answer vs correct
            prompt = f"Correct answer: {correct}\nUser answer: {ua}\nExplain briefly why the user's answer is correct or incorrect."
            try:
                expl = genai.generate_content(prompt).text
            except Exception:
                expl = ""
        else:
            expl = "Correct."
            correct_count += 1

        results.append({"question": stored_quiz[idx].get("question",""), "correct": correct, "user": ua, "is_correct": is_correct, "explanation": expl})
        explanations.append(expl)

    # Award points: e.g., 3 points per correct answer (you can change)
    points_per_correct = 3
    earned = correct_count * points_per_correct
    total_rewards += earned

    # Save evaluation document
    eval_id = f"eval_{uuid.uuid4().hex}"
    eval_doc = {"quiz_id": data.quiz_id, "results": results, "earned": earned, "time": time.time()}
    try:
        eval_collection.add(ids=[eval_id], documents=[json.dumps(eval_doc)])
    except Exception:
        pass

    # Save to history
    history_store.append({"id": eval_id, "type": "quiz_eval", "quiz_id": data.quiz_id, "results": results, "earned": earned, "time": time.time()})

    # Return summary
    return {"score": correct_count, "total": len(correct_answers), "earned": earned, "results": results}

# History endpoints
@app.get("/history")
def get_history():
    # return recent history (most recent first)
    return {"history": list(reversed(history_store))}

@app.post("/clear-history")
def clear_history():
    global history_store
    history_store = []
    return {"message": "History cleared!"}

# Rewards endpoint
@app.get("/rewards")
def get_rewards():
    return {"points": total_rewards}
