import hashlib
import os
from pathlib import Path
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import custom modules
from ai_engine import analyze_text
from database import FeedbackModel, UserModel, get_db, init_db

# Reliably locate the root directory containing main.py
BASE_DIR = Path(__file__).resolve().parent

# Initialize Database Tables
init_db()

app = FastAPI(
    title="SentimentShield API",
    description="An AI-powered feedback analysis API using FastAPI & SQLite",
    version="1.0.0",
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


class FeedbackRequest(BaseModel):
    text: str


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


# --- FRONTEND ROUTES ---


@app.get("/")
def read_root():
    # 1. First choice: sentiment-shield/templates/index.html
    templates_path = BASE_DIR / "templates" / "index.html"
    if templates_path.exists():
        return FileResponse(str(templates_path))

    # 2. Second choice: sentiment-shield/index.html
    root_path = BASE_DIR / "index.html"
    if root_path.exists():
        return FileResponse(str(root_path))

    # 3. Neither exists -> raise clear error
    raise HTTPException(
        status_code=404, 
        detail=f"index.html not found! Checked in '{templates_path.absolute()}' AND '{root_path.absolute()}'"
    )


@app.get("/login-page")
def read_login_page():
    # 1. First choice: sentiment-shield/templates/login.html
    templates_path = BASE_DIR / "templates" / "login.html"
    if templates_path.exists():
        return FileResponse(str(templates_path))

    # 2. Second choice: sentiment-shield/login.html
    root_path = BASE_DIR / "login.html"
    if root_path.exists():
        return FileResponse(str(root_path))

    # 3. Neither exists -> raise clear error
    raise HTTPException(
        status_code=404, 
        detail=f"login.html not found! Checked in '{templates_path.absolute()}' AND '{root_path.absolute()}'"
    )


@app.get("/api/health")
def health_check():
    return {"message": "SentimentShield API is up and running!"}


# --- API ENDPOINTS ---


@app.post("/analyze")
def analyze_and_store(request: FeedbackRequest, db: Session = Depends(get_db)):
    if not request.text.strip():
        raise HTTPException(
            status_code=400, detail="Text field cannot be empty."
        )

    ai_result = analyze_text(request.text)

    db_entry = FeedbackModel(
        text=request.text,
        sentiment=ai_result["sentiment"],
        polarity_score=ai_result["polarity_score"],
        urgency=ai_result["urgency"],
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)

    return {
        "status": "success",
        "data": {
            "id": db_entry.id,
            "text": db_entry.text,
            "sentiment": db_entry.sentiment,
            "polarity_score": db_entry.polarity_score,
            "urgency": db_entry.urgency,
        },
    }


@app.get("/logs")
def get_all_logs(db: Session = Depends(get_db)):
    logs = db.query(FeedbackModel).all()
    return {"total_logs": len(logs), "logs": logs}


@app.delete("/logs/{log_id}")
def delete_log(log_id: int, db: Session = Depends(get_db)):
    log_to_delete = (
        db.query(FeedbackModel).filter(FeedbackModel.id == log_id).first()
    )
    if not log_to_delete:
        raise HTTPException(status_code=404, detail="Log not found")

    db.delete(log_to_delete)
    db.commit()
    return {"message": f"Log #{log_id} deleted successfully"}


@app.post("/register")
def register(user: UserRegister, db: Session = Depends(get_db)):
    existing_user = (
        db.query(UserModel).filter(UserModel.username == user.username).first()
    )
    if existing_user:
        raise HTTPException(
            status_code=400, detail="Username already registered"
        )

    hashed_pw = hash_password(user.password)
    new_user = UserModel(username=user.username, password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully!"}


@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    hashed_pw = hash_password(user.password)
    db_user = (
        db.query(UserModel)
        .filter(
            UserModel.username == user.username, UserModel.password == hashed_pw
        )
        .first()
    )
    if not db_user:
        raise HTTPException(
            status_code=401, detail="Invalid username or password"
        )

    return {"message": "Login successful", "username": db_user.username}


@app.put("/logs/{log_id}/resolve")
def resolve_log(log_id: int, db: Session = Depends(get_db)):
    log_to_update = (
        db.query(FeedbackModel).filter(FeedbackModel.id == log_id).first()
    )
    if not log_to_update:
        raise HTTPException(status_code=404, detail="Log not found")

    current_status = getattr(log_to_update, "status", "Pending") or "Pending"
    log_to_update.status = (
        "Pending" if current_status == "Resolved" else "Resolved"
    )

    db.commit()
    db.refresh(log_to_update)

    return {
        "message": f"Log #{log_id} status updated",
        "status": log_to_update.status,
    }