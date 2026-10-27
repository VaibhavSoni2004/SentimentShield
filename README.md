# ComplaintFlow (SentimentShield)

> An intelligent, hybrid complaint triage and feedback analytics platform combining Google Gemini structured JSON outputs with an offline, rule-based NLP fallback engine.

---

> [!NOTE]
> **Educational & Demonstration Notice**: ComplaintFlow is an educational and portfolio demonstration project. While it includes urgency classification heuristics for hazards, outages, and financial fraud, **urgent real-world physical safety emergencies or medical incidents must always be directed immediately to official emergency response services (e.g., 911, local emergency hotlines).**

---

## 1. Overview & Problem Statement

Customer support and operational departments frequently handle hundreds of unstructured complaint messages daily. In conventional support workflows, sorting through high-volume queues manually delays the identification of critical incidents—such as security breaches, utility cuts, or active safety hazards—while cosmetic issues or false alarms consume valuable response time.

**ComplaintFlow** addresses this operational challenge with a dual-dimension triage system:
1. **Decoupled Sentiment and Urgency Analysis**: The system evaluates both emotional tone (**Sentiment**: *Positive*, *Negative*, *Neutral* with a polarity score from -1.0 to +1.0) and practical impact (**Operational Urgency**: *High*, *Medium*, *Low* with a severity score from 5 to 99). Highly negative customer venting about a minor issue is correctly distinguished from an urgent, calmly reported infrastructure failure.
2. **Resilient Hybrid AI Pipeline**: Primary classification uses Google's **Gemini 3.5 Flash Lite** model with strict Pydantic JSON schema enforcement. If the external Gemini API experiences transient errors, timeouts, or quota limits, an offline deterministic NLP engine (VADER + domain heuristics) serves as a fallback, allowing the application to continue processing feedback locally.
3. **Role-Based Access Control (RBAC)**: Secure JWT authentication ensures that customers can submit complaints and track only their personal submissions, while administrators oversee system-wide logs, update resolution statuses, and manage records.

---

## 2. Architecture & Hybrid Pipeline

```
                              [ User Complaint ]
                                      |
                                      v
                         [ POST /analyze (JWT Auth) ]
                                      |
               +----------------------+----------------------+
               |                                             |
               v (Primary)                                   v (Fallback on Failure)
   [ Google Gemini API ]                            [ Rule-Based NLP Engine ]
- Model: gemini-3.5-flash-lite                   - NLTK VADER Compound Polarity
- Schema: ComplaintAnalysis (Pydantic)           - Keyword Dictionaries & Regex
- thinking_level: LOW                            - Negation & Drill Filters
- 1 Bounded Retry (1.0s delay)                   - Deterministic Score Assignment
               |                                             |
               +----------------------+----------------------+
                                      |
                                      v
                     [ validate_and_align_urgency() ]
                     - Enforces Category-Score Bounds
                     - High: 70-99 | Med: 40-69 | Low: 5-39
                                      |
                                      v
                   [ SQLite Database (feedback.db) ]
                                      |
                                      v
                          [ Structured JSON Response ]
```

### Primary Analyzer: Google Gemini
- **Model**: `gemini-3.5-flash-lite` (configurable via `GEMINI_MODEL`).
- **Low-Latency Thinking**: For Gemini 3 models, `thinking_level=LOW` is attached via `types.ThinkingConfig` to minimize response latency while maintaining structured classification reasoning.
- **Structured Output**: Enforces strict schema conformance using a Pydantic `ComplaintAnalysis` model (`response_mime_type="application/json"` and `response_schema=ComplaintAnalysis`).
- **Bounded Retry for Transient Errors**: If a request encounters a transient failure (HTTP 429, 500, 502, 503, 504, transport timeout, or connection drop), the engine waits 1.0 second and retries **at most once**. Permanent errors (HTTP 400, 401, 403, 404, missing API keys, or schema validation errors) fail fast without retry.

### Fallback Analyzer: Deterministic NLP Engine
If all Gemini attempts fail or if `GEMINI_API_KEY` is omitted, the engine automatically invokes `fallback_analyze_text()`:
- **Sentiment**: Calculated using NLTK's `SentimentIntensityAnalyzer` (VADER) compound score (-1.0 to +1.0).
- **Urgency Heuristics**: Evaluated across specialized regex domain dictionaries covering:
  - *High Urgency*: Gas leaks, fire/electrical hazards, life-threatening situations, unauthorized bank transfers/theft, complete server outages.
  - *Medium Urgency*: Utility outages (power, water, sewage), ATM failures, application crashes, account lockouts, billing discrepancies.
  - *Low Urgency*: Routine maintenance, cosmetic UI glitches, typos, feature suggestions, general inquiries.
  - *Contextual Modifiers*: Explicit negations (e.g., *"not hacked"*, *"no flood"*), scheduled drills, and historical/resolved incidents (*"resolved yesterday"*) are systematically downgraded to Low urgency.

---

## 3. Urgency Bands & Metric Alignment

To prevent discrepancies between numerical urgency scores and urgency categories, the application enforces the following strict boundaries via `validate_and_align_urgency()`:

| Urgency Level | Score Range | Operational Definition & Criteria |
| :--- | :---: | :--- |
| **High** | **70 – 99** | Immediate physical hazard, active financial theft/fraud, complete infrastructure collapse, or critical security breach requiring urgent human response. |
| **Medium** | **40 – 69** | Disrupted core operations, utility failures, account access lockouts, software service crashes, or significant customer service failure. |
| **Low** | **5 – 39** | Informational queries, cosmetic improvements, typos, routine maintenance, scheduled safety drills, or negated/resolved past events. |

---

## 4. Key Features

- **Decoupled Classification**: Urgency is separated from sentiment; negative emotion does not automatically imply operational urgency.
- **Role-Based Isolation**:
  - **Standard Users**: Register, sign in, submit complaints, view real-time score breakdowns, and view their personal submission history.
  - **Administrators**: Access system-wide records, filter complaints by urgency or category, search feedback, toggle ticket status (`Pending` ↔ `Resolved`), and delete entries.
- **Safe Diagnostic Logging**: Error messages are sanitized using regex redaction (`re.sub(r"(key|token|auth|secret)=[\w-]+", ...)`) to strip potential credential leaks, while transport and network exceptions emit static diagnostic strings.
- **Production Secret Guard**: On application startup, a fail-fast check ensures `SECRET_KEY` is explicitly set to a custom value when `ENVIRONMENT=production`, rejecting missing values and the default `'dev-secret-key'`.
- **Client Session Integrity**: The frontend checks JWT token expiration (`isTokenExpired`) on page load and before fetch calls, clearing local credentials on 401 Unauthorized to prevent infinite redirect loops.

---

## 5. Tech Stack

- **Backend Framework**: [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Server**: [Uvicorn](https://www.uvicorn.org/)
- **AI & NLP**:
  - [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python) (`google-genai >= 1.0.0`)
  - [NLTK](https://www.nltk.org/) (VADER Sentiment Lexicon)
  - [Pydantic v2](https://docs.pydantic.dev/) (Strict JSON validation)
- **Database & ORM**: [SQLAlchemy](https://www.sqlalchemy.org/) with SQLite (`feedback.db`)
- **Authentication**: JWT via [python-jose](https://github.com/mpdavis/python-jose) (HS256) and SHA-256 password hashing
- **Frontend**: HTML5, Vanilla JavaScript (ES6+), Tailwind CSS (via CDN)

---

## 6. Project Structure

```
SentimentShield/
├── README.md                      # Project documentation (Git repository root)
└── sentiment-shield/              # Core application directory
    ├── main.py                    # FastAPI entry point, routing, RBAC dependencies, static file serving
    ├── ai_engine.py               # Hybrid AI engine (Gemini API, retries, VADER fallback, heuristics)
    ├── database.py                # SQLite database engine, SQLAlchemy session setup, UserModel & FeedbackModel
    ├── requirements.txt           # Project dependencies
    ├── test_classification.py     # 43 offline unit tests (classification, Gemini mocks, auth, safety guards)
    ├── .gitignore                 # Excludes .env, local feedback.db, virtual environments, and caches
    └── templates/
        ├── index.html             # Customer and Admin dashboard (submission, live cards, filtered table)
        └── login.html             # Authentication portal (user/admin login and registration)
```

---

## 7. Setup & Installation (Windows PowerShell)

Follow these steps for a clean clone and setup on Windows:

### 1. Clone the repository and navigate to the application folder
```powershell
git clone https://github.com/VaibhavSoni2004/SentimentShield.git
cd SentimentShield\sentiment-shield
```

### 2. Create and activate a virtual environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> [!TIP]
> If script execution is blocked on Windows, run the following command in your PowerShell session before activating:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
> ```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
```
*(Note: NLTK's `vader_lexicon` is automatically verified and downloaded by `ai_engine.py` on first run if not already cached).*

---

## 8. Environment Configuration (`.env`)

Create a `.env` file inside the `sentiment-shield/` directory. **Use placeholder values only and never commit this file to version control:**

```ini
# Environment mode: 'development' or 'production'
# In production, the application refuses to start if SECRET_KEY is missing or set to 'dev-secret-key'.
ENVIRONMENT=development

# JWT Secret Key: In production, generate a secure 64-character hex string:
# python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=dev-secret-key

# JWT Token expiration in minutes (default: 1440 = 24 hours)
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Google Gemini API Configuration (optional for offline fallback, required for cloud AI triage)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_TIMEOUT_MS=60000

# Initial Admin Bootstrap Account (auto-created on startup if configured)
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_secure_admin_password_here
```

---

## 9. Running the Application

From within the `sentiment-shield/` directory, start the local development server:

```powershell
uvicorn main:app --reload --port 8000
```

Once running:
- **Web Application / Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Login / Register Page**: Open [http://127.0.0.1:8000/login-page](http://127.0.0.1:8000/login-page)
- **Interactive API Documentation (Swagger UI)**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check**: Open [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 10. Running the Test Suite

The test suite runs entirely offline using mocks and does not consume external Gemini API quota.

From within the `sentiment-shield/` directory, run the test suite with verbose output:

```powershell
python -m unittest test_classification.py -v
```

> **Test Suite Status**: **43 unit tests passed** in the last reported verification run.

### Test Coverage Highlights:
- **`TestClassificationRegressions`** (11 tests): Verifies heuristic accuracy for financial fraud, physical hazards, utility cuts, emotional venting, contextual negations, and scheduled drills.
- **`TestGeminiIntegration`** (18 tests): Verifies structured Pydantic parsing, `thinking_level=LOW` inclusion, single-retry execution on transient 503 errors, immediate fail-fast on 400/403 client errors, and fallback activations on 504 server deadlines, transport timeouts, or missing API keys.
- **`TestAuthenticationFlow`** (6 tests): Verifies token encoding/decoding, expiration handling (401), missing claim rejection, and role-based route access controls.
- **`TestSecretKeyProductionSafety`** (8 tests): Verifies fail-fast startup behavior in production when `SECRET_KEY` is missing, empty, or left as `'dev-secret-key'`.

---

## 11. API Reference

| Method | Endpoint | Access Level | Description |
| :--- | :--- | :---: | :--- |
| `GET` | `/` | Public | Serves the main complaint dashboard (`templates/index.html`). |
| `GET` | `/login-page` | Public | Serves the user and administrator authentication interface (`templates/login.html`). |
| `GET` | `/api/health` | Public | Health check endpoint returning operational status. |
| `POST` | `/register` | Public | Registers a new user account with `role="user"`. |
| `POST` | `/login` | Public | Authenticates credentials, validates requested role, and returns a JWT access token. |
| `POST` | `/token` | Public | OAuth2 password request form endpoint for Swagger UI authorization (`/docs`). |
| `POST` | `/analyze` | Authenticated | Submits feedback text, executes hybrid triage, persists the record, and returns analysis. |
| `GET` | `/logs` | Authenticated | Retrieves logs. Regular users receive only their own submissions; admins receive all submissions. |
| `PUT` | `/logs/{log_id}/resolve` | Admin Only | Toggles the resolution status of a complaint between `Pending` and `Resolved`. |
| `DELETE` | `/logs/{log_id}` | Admin Only | Permanently deletes a feedback log entry by ID. |

---

## 12. Security Considerations

1. **Production Secret Enforcement**: If `ENVIRONMENT=production` (or `prod`), [`main.py`](sentiment-shield/main.py) immediately halts application startup with a `RuntimeError` if `SECRET_KEY` is missing, whitespace, or set to `'dev-secret-key'`.
2. **Sanitized Exception Logging**: In `_format_safe_gemini_error()`, API error messages are scrubbed using regular expression replacement (`re.sub(r"(key|token|auth|secret)=[\w-]+", r"\1=[REDACTED]", msg)`) to prevent credentials from appearing in application logs. Network and transport timeouts emit static diagnostic messages.
3. **Role-Based Isolation (RBAC)**: Enforced through FastAPI dependency injection (`get_current_user` and `get_current_admin`). Admin-only capabilities (`PUT /logs/{id}/resolve` and `DELETE /logs/{id}`) reject non-admin users with an HTTP 403 Forbidden status.
4. **Credential Isolation**: Local database files (`feedback.db`), environment files (`.env`), and cache directories are excluded from version control via `.gitignore`.

---

## 13. Current Limitations & Future Improvements

- **Password Hashing**: Passwords are currently stored using raw SHA-256 (`hashlib.sha256`) without per-user salts. For production systems, migration to a dedicated password-hashing algorithm such as bcrypt (which is already included in `requirements.txt` via `bcrypt==5.0.0` and `passlib==1.7.4`) or Argon2id is strongly recommended.
- **Database Engine**: The application currently uses SQLite (`feedback.db`). While suitable for local development and single-process instances, a concurrent multi-worker production environment should adopt a client-server relational database such as PostgreSQL.
- **Log Pagination**: Pagination is currently not implemented. While the frontend provides real-time client-side search, category filtering, urgency filtering, and sorting, all matching records are rendered at once. Introducing database-level `LIMIT` and `OFFSET` parameters will optimize performance for large datasets.
- **Heuristic Scope**: The fallback classification engine is rule-based and should be calibrated and evaluated against domain-specific datasets before real-world reliance.
- **Admin Provisioning**: Initial administrator credentials are read from `.env` at module startup. Decoupling bootstrap account creation into a dedicated CLI command or database migration would improve administrative lifecycle management.
