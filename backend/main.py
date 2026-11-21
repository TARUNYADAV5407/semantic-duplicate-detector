import io
import re
from typing import List
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pypdf import PdfReader
import docx

# Load lightweight quantized ONNX model & fast Rust tokenizer
tokenizer = Tokenizer.from_file("onnx_model/tokenizer.json")
tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
tokenizer.enable_truncation(max_length=128)
session = ort.InferenceSession("onnx_model/model_quantized.onnx", providers=["CPUExecutionProvider"])

app = FastAPI(title="Fine-Tuned Semantic Duplicate Question Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


class TextRequest(BaseModel):
    text: str
    threshold: float = 0.75


def encode_questions(questions: List[str]) -> np.ndarray:
    encoded = tokenizer.encode_batch(questions)
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encoded], dtype=np.int64)

    outputs = session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )[0]

    # Mean pooling + L2 normalization
    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
    pooled = (outputs * mask).sum(axis=1) / np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.clip(norms, a_min=1e-9, a_max=None)


def extract_questions(raw_text: str) -> List[str]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    q_pattern = re.compile(r"^(?:Q\.?\s*\d+[\.\):]?|\d+[\.\)]|\(\d+\))\s+", re.IGNORECASE)

    if not any(q_pattern.match(line) for line in lines):
        return [line for line in lines if len(line.split()) >= 4]

    questions, current_q = [], ""
    for line in lines:
        if q_pattern.match(line):
            if current_q:
                questions.append(current_q.strip())
            current_q = q_pattern.sub("", line).strip()
        elif current_q:
            current_q += " " + line

    if current_q:
        questions.append(current_q.strip())
    return [q for q in questions if len(q.split()) >= 3]


def detect_duplicates(questions: List[str], threshold: float):
    n = len(questions)
    if n < 2:
        return {
            "total_questions": n,
            "duplicate_pairs": [],
            "unique_questions": questions,
            "duplicates_removed": 0,
        }

    embeddings = encode_questions(questions)
    cosine_scores = np.matmul(embeddings, embeddings.T)

    duplicate_pairs = []
    duplicate_indices = set()

    for i in range(n):
        for j in range(i + 1, n):
            score = float(cosine_scores[i, j])
            if score >= threshold:
                duplicate_pairs.append({
                    "q1_index": i + 1,
                    "q1_text": questions[i],
                    "q2_index": j + 1,
                    "q2_text": questions[j],
                    "similarity": round(max(0.0, score) * 100, 2),
                })
                duplicate_indices.add(j)

    duplicate_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    unique_questions = [q for idx, q in enumerate(questions) if idx not in duplicate_indices]

    return {
        "total_questions": n,
        "duplicate_pairs": duplicate_pairs,
        "unique_questions": unique_questions,
        "duplicates_removed": len(duplicate_indices),
    }


@app.post("/analyze-text")
async def analyze_text(req: TextRequest):
    questions = extract_questions(req.text)
    if not questions:
        raise HTTPException(status_code=400, detail="No valid questions found in the text.")
    return detect_duplicates(questions, req.threshold)


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...), threshold: float = Form(0.75)):
    filename = file.filename.lower()
    content = await file.read()
    if filename.endswith(".txt"):
        raw_text = content.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(content))
        raw_text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    questions = extract_questions(raw_text)
    if not questions:
        raise HTTPException(status_code=400, detail="Could not extract any questions from the file.")
    return detect_duplicates(questions, threshold)