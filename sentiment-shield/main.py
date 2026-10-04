import hashlib
import os
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

# Import custom modules
from ai_engine import analyze_text
from database import FeedbackModel, UserModel, get_db, init_db

# Reliably locate the root directory containing main.py
BASE_DIR = Path(__file__).resolve().parent

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# Initialize Database Tables
init_db()

app = FastAPI(
    title="ComplaintFlow API",
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

def create_default_admin():
    admin_username = os.getenv("ADMIN_USERNAME")
    admin_password = os.getenv("ADMIN_PASSWORD")

    if not admin_username or not admin_password:
        return

    db = next(get_db())

    try:
        existing_admin = (
            db.query(UserModel)
            .filter(UserModel.username == admin_username)
            .first()
        )

        if not existing_admin:
            admin = UserModel(
                username=admin_username,
                password=hash_password(admin_password),
                role="admin"
            )
            db.add(admin)
            db.commit()
            print(f"Default admin created: {admin_username}")

        elif existing_admin.role != "admin":
            existing_admin.role = "admin"
            db.commit()
            print(f"Existing account promoted to admin: {admin_username}")

    finally:
        db.close()

create_default_admin()

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )

    to_encode.update({"exp": expire})

    return jwt.encode(
        to_encode,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM]
        )

        username = payload.get("sub")
        role = payload.get("role")

        if username is None or role is None:
            raise credentials_exception

        return {
            "username": username,
            "role": role
        }

    except JWTError:
        raise credentials_exception

def get_current_admin(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )

    return current_user

class FeedbackRequest(BaseModel):
    text: str


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str
    login_type: str = "user"


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
        urgency_score=ai_result["urgency_score"],
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
            "urgency_score": db_entry.urgency_score,
        },
    }


@app.get("/logs")
def get_all_logs(db: Session = Depends(get_db)):
    logs = db.query(FeedbackModel).all()
    return {"total_logs": len(logs), "logs": logs}


@app.delete("/logs/{log_id}")
def delete_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_admin: dict = Depends(get_current_admin)
):
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
            UserModel.username == user.username,
            UserModel.password == hashed_pw
        )
        .first()
    )

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    # Verify that the selected login type matches
    # the actual role stored in the database.
    if user.login_type == "admin" and db_user.role != "admin":
        raise HTTPException(
            status_code=403,
            detail="This account does not have administrator privileges"
        )

    if user.login_type == "user" and db_user.role != "user":
        raise HTTPException(
            status_code=403,
            detail="Please use Admin Login for this account"
        )

    access_token = create_access_token(
        data={
            "sub": db_user.username,
            "role": db_user.role
        }
    )

    return {
        "message": "Login successful",
        "username": db_user.username,
        "role": db_user.role,
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.post("/token")
def login_for_swagger(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    hashed_pw = hash_password(form_data.password)

    db_user = (
        db.query(UserModel)
        .filter(
            UserModel.username == form_data.username,
            UserModel.password == hashed_pw
        )
        .first()
    )

    if not db_user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    access_token = create_access_token(
        data={
            "sub": db_user.username,
            "role": db_user.role
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@app.put("/logs/{log_id}/resolve")
def resolve_log(
    log_id: int,
    db: Session = Depends(get_db),
    current_admin: dict = Depends(get_current_admin)
):
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