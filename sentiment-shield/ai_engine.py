import logging
import os
import re
import time
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
import httpx
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Safely load environment variables from local .env if present
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# -----------------------------
# VADER Setup for Fallback
# -----------------------------
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()


# -----------------------------
# Gemini Exceptions & Models
# -----------------------------
class GeminiConfigurationError(Exception):
    """Raised when the GEMINI_API_KEY environment variable is missing or invalid."""
    pass


class ComplaintAnalysis(BaseModel):
    """Structured output schema for Gemini complaint analysis."""
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(
        description="Sentiment classification: Positive, Negative, or Neutral."
    )
    polarity_score: float = Field(
        description="Sentiment polarity score from -1.0 (most negative) to 1.0 (most positive)."
    )
    urgency: Literal["High", "Medium", "Low"] = Field(
        description="Operational urgency level: High, Medium, or Low based on severity and immediate action required."
    )
    urgency_score: int = Field(
        description="Operational urgency score from 5 to 99 representing severity, NOT sentiment alone."
    )

    @field_validator("polarity_score")
    @classmethod
    def clamp_polarity(cls, v: float) -> float:
        return round(max(-1.0, min(1.0, float(v))), 2)

    @field_validator("urgency_score")
    @classmethod
    def clamp_urgency_score(cls, v: int) -> int:
        return max(5, min(99, int(v)))


def validate_and_align_urgency(urgency: str, urgency_score: int) -> tuple[str, int]:
    """Ensures strict consistency between urgency category and score ranges:
    - High: 70 to 99
    - Medium: 40 to 69
    - Low: 5 to 39
    """
    urgency_score = max(5, min(99, int(urgency_score)))
    if urgency == "High":
        if urgency_score < 70:
            urgency_score = max(70, urgency_score)
    elif urgency == "Medium":
        if urgency_score < 40 or urgency_score >= 70:
            urgency_score = max(40, min(69, urgency_score))
    elif urgency == "Low":
        if urgency_score >= 40:
            urgency_score = min(39, urgency_score)
    else:
        # Default category derived from score if an unknown label is provided
        if urgency_score >= 70:
            urgency = "High"
        elif urgency_score >= 40:
            urgency = "Medium"
        else:
            urgency = "Low"

    return urgency, urgency_score


GEMINI_SYSTEM_INSTRUCTION = (
    "You are an expert complaint analysis assistant for an enterprise complaint management system.\n"
    "Analyze the user's complaint text and determine:\n"
    "1. sentiment: 'Positive', 'Negative', or 'Neutral'.\n"
    "2. polarity_score: Float between -1.0 (most negative) and +1.0 (most positive).\n"
    "3. urgency: 'High', 'Medium', or 'Low'.\n"
    "4. urgency_score: Integer between 5 and 99.\n\n"
    "CRITICAL RULES:\n"
    "- Urgency represents operational urgency, safety hazard, financial theft, or infrastructure failure.\n"
    "- Urgency is NOT sentiment. Negative sentiment alone does NOT imply high urgency.\n"
    "- High (70-99): Immediate physical danger, gas leaks, electrical hazards, life-threatening medical emergencies, "
    "unauthorized bank transfers/fraud, complete server outages, active disaster alerts.\n"
    "- Medium (40-69): Functional disruptions, utility outages (power/water/sewage cuts), ATM issues, app crashes, "
    "account lockout, delayed services, rude customer support.\n"
    "- Low (5-39): Routine maintenance, informational requests, feature suggestions, minor cosmetic issues (typos, button colors), "
    "resolved past incidents, false alarms, cancelled warnings.\n"
    "- Negations: If an emergency is negated (e.g. 'not hacked', 'no flood', 'false alarm'), classify as Low urgency.\n"
    "- Historical/Resolved: If an incident is already resolved or in the past (e.g. 'flood receded last week and road reopened'), "
    "classify as Low urgency."
)


def _get_effective_timeout(client: genai.Client | None = None) -> int:
    """Safely extracts the effective timeout in milliseconds from client or environment.

    Guarantees no secrets or headers are read or exposed.
    """
    if client is not None:
        api_client = getattr(client, "_api_client", None)
        http_options = getattr(api_client, "_http_options", None)
        timeout = getattr(http_options, "timeout", None)
        if isinstance(timeout, (int, float)):
            return int(timeout)
    try:
        return int(os.getenv("GEMINI_TIMEOUT_MS", "10000"))
    except (ValueError, TypeError):
        return 10000


def _supports_thinking_level(model_name: str) -> bool:
    """Checks whether the configured model supports thinking_level configuration.

    Gemini 3 series (e.g. gemini-3.5-flash-lite, gemini-3.8-flash) and Gemini 2.5 series models support
    thinking_level controls ('low', 'medium', 'high').
    """
    clean_model = model_name.lower().removeprefix("models/")
    return clean_model.startswith(("gemini-3", "gemini-2.5"))


TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}


def _is_transient_error(exc: Exception) -> bool:
    """Determines whether an exception is a transient error eligible for retry.

    Retries transient service and rate-limit errors: HTTP 429, 500, 502, 503, 504,
    as well as transient network timeouts and connection drops.
    Permanent errors (400, 401, 403, 404, ValidationError, GeminiConfigurationError) are NOT retried.
    """
    if isinstance(exc, (GeminiConfigurationError, ValueError)):
        return False

    if isinstance(exc, APIError):
        code = getattr(exc, "code", None)
        if code in TRANSIENT_HTTP_STATUS_CODES:
            return True
        if code is not None and (400 <= code < 500):
            return False
        if code is not None and (500 <= code < 600):
            return True

    if isinstance(exc, (httpx.TimeoutException, TimeoutError, httpx.ConnectError)):
        return True

    return False


def _get_gemini_client() -> genai.Client:
    """Instantiates the official Google GenAI client securely from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise GeminiConfigurationError("GEMINI_API_KEY environment variable is missing or empty.")

    timeout_ms = int(os.getenv("GEMINI_TIMEOUT_MS", "10000"))
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(timeout=timeout_ms)
    )


def _analyze_with_gemini(text: str, client: genai.Client | None = None) -> dict:
    """Primary analyzer using Google Gemini API with structured JSON output."""
    if client is None:
        client = _get_gemini_client()

    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    effective_timeout_ms = _get_effective_timeout(client)

    thinking_config = None
    if _supports_thinking_level(model_name):
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.LOW,
        )

    max_attempts = 2  # 1 initial attempt + at most 1 bounded retry
    response = None
    last_exc = None

    for attempt in range(1, max_attempts + 1):
        try:
            t0 = time.perf_counter()
            response = client.models.generate_content(
                model=model_name,
                contents=text,
                config=types.GenerateContentConfig(
                    system_instruction=GEMINI_SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=ComplaintAnalysis,
                    temperature=0.0,
                    thinking_config=thinking_config,
                ),
            )
            elapsed = time.perf_counter() - t0
            logger.info(
                f"Gemini API call succeeded in {elapsed:.2f}s "
                f"(model={model_name}, effective_timeout_ms={effective_timeout_ms}, attempt={attempt})."
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts and _is_transient_error(exc):
                diag = _format_safe_gemini_error(exc)
                logger.info(
                    f"Gemini API call encountered transient error ({diag}); "
                    f"retrying in 1.0s (attempt {attempt} of {max_attempts - 1} retry)..."
                )
                time.sleep(1.0)
                continue
            raise last_exc

    if not response:
        raise ValueError("Received empty response from Gemini API.")

    # Extract parsed model or validate json text
    if hasattr(response, "parsed") and isinstance(response.parsed, ComplaintAnalysis):
        analysis = response.parsed
    elif hasattr(response, "text") and response.text:
        analysis = ComplaintAnalysis.model_validate_json(response.text)
    else:
        raise ValueError("Could not extract valid analysis data from Gemini response.")

    urgency, urgency_score = validate_and_align_urgency(analysis.urgency, analysis.urgency_score)

    return {
        "sentiment": analysis.sentiment,
        "polarity_score": round(analysis.polarity_score, 2),
        "urgency": urgency,
        "urgency_score": urgency_score,
    }


# -----------------------------
# Contextual Modifier Patterns (Fallback)
# -----------------------------
CANCELLATION_OR_DRILL_PATTERN = re.compile(
    r"\b(false alarm|mock drill|test alert|drill only|practice drill|evacuation drill|"
    r"warning\s+(has been\s+|is\s+)?cancelled|alert\s+(has been\s+|is\s+)?withdrawn)\b",
    re.IGNORECASE
)

RESOLVED_OR_PAST_PATTERN = re.compile(
    r"\b(already\s+(resolved|fixed|repaired|restored|reopened|cleared)|"
    r"receded|back to normal|completely fine now|fully reopened|"
    r"resolved\s+(yesterday|last week|earlier)|problem\s+(is\s+|has been\s+)?(solved|resolved)|"
    r"as requested|i authorized|requested by me|it was just my\s+\w+)\b",
    re.IGNORECASE
)

COSMETIC_AESTHETIC_PATTERN = re.compile(
    r"\b(logo font|new logo|ugly logo|color choice|ugly color|font size|button color|"
    r"color scheme|dark mode toggle|dark theme icon|misaligned|spelling typo)\b",
    re.IGNORECASE
)


# -----------------------------
# Domain-Specific Urgency Patterns (Fallback)
# -----------------------------
HIGH_URGENCY_REGEXES = [
    # Natural disasters & public safety hazards
    re.compile(
        r"\b(flood|flash flood|wildfire|forest fire|fire outbreak|building on fire|"
        r"gas leak|toxic leak|chemical spill|chemical tanker|hazardous acid|collaps\w*|"
        r"dam breach|explosion|bomb threat|active shooter|disaster|hazard|danger|emergency)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b((live|snapped|sparking|exposed)\s+(electrical\s+)?wire|electric shock hazard|fallen (electric )?pole)\b",
        re.IGNORECASE
    ),
    # Medical & personal life safety
    re.compile(
        r"\b((cannot|can't|can\s+not|unable to)\s+breath\w*|severe chest pain|heart attack|"
        r"cardiac arrest|stroke|unconscious|collapsed on the floor|unresponsive|"
        r"profuse bleeding|heavy bleeding|swallowed toxic|swallowed poison|send ambulance)\b",
        re.IGNORECASE
    ),
    re.compile(
        r"\b(power outage in\s+(hospital|icu|ventilator|operation theatre|emergency ward))\b",
        re.IGNORECASE
    ),
    # Financial crime, unauthorized theft, and account compromise
    re.compile(
        r"\b(unauthorized\s+(international\s+)?(transaction|transfer|withdrawal|charge)|"
        r"transferred\s+[\w\s,]{0,60}?\bwithout\s+otp|account\s+(was\s+)?(hacked|compromised|frozen|blocked)|"
        r"account\s+(frozen|blocked)|all (my )?savings (were )?drained|identity theft|"
        r"ransomware|data breach|extortion|fraud|scam|stolen|money stolen|hacked)\b",
        re.IGNORECASE
    ),
    # Critical system/infrastructure outage
    re.compile(
        r"\b(complete\s+(production\s+)?(server\s+)?outage|all services down( for all users)?|"
        r"water pipeline burst.*(flood|sewage))\b",
        re.IGNORECASE
    ),
    # Regulatory & critical escalation
    re.compile(
        r"\b(critical escalation|regulatory notice|ombudsman|legal notice)\b",
        re.IGNORECASE
    ),
    # Explicit high risk indicators
    re.compile(
        r"\b(high[- ]risk(\s+alert)?|critical (hazard|risk|danger)|emergency assistance)\b",
        re.IGNORECASE
    ),
]

MEDIUM_URGENCY_REGEXES = [
    # Banking inconveniences & transaction disputes
    re.compile(
        r"\b(atm swallowed|money debited.*(not (received|credited)|did not receive)|"
        r"debited.*not (received|credited)|payment.*(failed|not received)|double deduct\w*|"
        r"double debit\w*|card blocked|payment gateway\s+500|debited|frozen|hijacked)\b",
        re.IGNORECASE
    ),
    # Utilities disruptions & infrastructure failures
    re.compile(
        r"\b(water (supply )?disrupted|no water supply|sewage (drain )?overflow\w*|"
        r"street lights? not working|traffic (lights?|signal) not working|power outage|"
        r"blackout|transformer burnt)\b",
        re.IGNORECASE
    ),
    # Digital / App issues
    re.compile(
        r"\b(application crashed|app crashed|crashed repeatedly|crashed|"
        r"cannot reset (my )?password|verification link expired|otp not received|"
        r"login (failed|not working|is not working)|is not working|not working|failed|500 error)\b",
        re.IGNORECASE
    ),
    # Service escalations & delays
    re.compile(
        r"\b(pending for \d+\s+(day|week|month)s?|no response (for|since)|"
        r"ticket\s+#?\d+\s+has been pending|rude (and unhelpful )?(executive|agent|support)|"
        r"unresolved complaint|delivery package was delayed|delayed by \d+\s+days)\b",
        re.IGNORECASE
    ),
    # Explicit medium risk & warnings
    re.compile(
        r"\b(medium[- ]risk|warning|caution)\b",
        re.IGNORECASE
    ),
]

LOW_URGENCY_REGEXES = [
    # Routine maintenance & service requests
    re.compile(
        r"\b(routine (office )?cleaning|routine (health )?checkup|routine tree pruning|"
        r"scheduled maintenance|garbage collection|pothole|asphalt repair)\b",
        re.IGNORECASE
    ),
    # Informational & statements
    re.compile(
        r"\b(pricing tiers|enterprise plans|send (my )?quarterly bank account statement|"
        r"account statement by email)\b",
        re.IGNORECASE
    ),
    # Suggestions & minor cosmetic feedback
    re.compile(
        r"\b(dark mode toggle|misaligned|spelling typo|grammar typo|small ui suggestion|"
        r"feature request|would be nice|in future updates?|add support for|"
        r"slow|suggestion|feedback|nice|great)\b",
        re.IGNORECASE
    ),
    # Explicit low risk
    re.compile(
        r"\b(low[- ]risk|cosmetic glitch|minor (issue|glitch|bug|typo))\b",
        re.IGNORECASE
    ),
]


def is_negated_match(text: str, match_span) -> bool:
    """Checks whether an emergency/hazard/fraud keyword is negated in the immediate clause."""
    matched_word = text[match_span[0]:match_span[1]].lower()
    # 'not working', 'failed', 'crashed' are failure indicators, not negations of failure
    if "not working" in matched_word or "failed" in matched_word or "crashed" in matched_word:
        return False

    start = match_span[0]
    window_start = max(0, start - 35)
    preceding_text = text[window_start:start].lower()
    last_clause = re.split(r"[.!?;\n]", preceding_text)[-1]

    return bool(
        re.search(
            r"\b(not|no|never|neither|n't|wasn't|isn't|aren't|didn't|doesn't|without any|fake|untrue)\s*$",
            last_clause.strip(),
        )
        or re.search(
            r"\b(not|no|never|neither|n't|wasn't|isn't|aren't|didn't|doesn't|without any|fake|untrue)\s+\w+(\s+\w+)?$",
            last_clause.strip(),
        )
    )


def fallback_analyze_text(text: str) -> dict:
    """Deterministic fallback analyzer using NLTK VADER and domain regex heuristics."""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # 1. Context Analysis: Cancellations, Mock Drills, Resolved Incidents, Cosmetic Only
    is_drill_or_cancelled = bool(CANCELLATION_OR_DRILL_PATTERN.search(text))
    is_resolved = bool(RESOLVED_OR_PAST_PATTERN.search(text))
    is_cosmetic_only = bool(COSMETIC_AESTHETIC_PATTERN.search(text))

    # 2. Check High Urgency Patterns (filtering out negated occurrences)
    high_matched = False
    for regex in HIGH_URGENCY_REGEXES:
        for match in regex.finditer(text):
            if not is_negated_match(text, match.span()):
                high_matched = True
                break
        if high_matched:
            break

    # 3. Check Medium Urgency Patterns (filtering out negated occurrences)
    medium_matched = False
    for regex in MEDIUM_URGENCY_REGEXES:
        for match in regex.finditer(text):
            if not is_negated_match(text, match.span()):
                medium_matched = True
                break
        if medium_matched:
            break

    # 4. Check Low Urgency Patterns
    low_matched = any(bool(regex.search(text)) for regex in LOW_URGENCY_REGEXES)

    # 5. Base Urgency from Sentiment
    if compound < 0:
        base_urgency = 50 + (abs(compound) * 45)
    else:
        base_urgency = max(5, 30 - (compound * 25))

    # 6. Determine Urgency Category and Urgency Score
    if is_drill_or_cancelled or is_resolved:
        urgency = "Low"
        urgency_score = min(30, max(10, round(base_urgency * 0.4)))
    elif high_matched:
        urgency = "High"
        urgency_score = max(80, min(98, round(max(85, base_urgency))))
    elif medium_matched:
        urgency = "Medium"
        urgency_score = max(45, min(68, round(max(50, min(65, base_urgency)))))
    elif is_cosmetic_only or low_matched:
        urgency = "Low"
        urgency_score = min(35, max(10, round(min(30, base_urgency))))
    else:
        # Fallback when no domain patterns match (general unstructured feedback)
        if base_urgency >= 75:
            urgency = "High"
            urgency_score = min(90, max(72, round(base_urgency)))
        elif base_urgency >= 40:
            urgency = "Medium"
            urgency_score = min(68, max(42, round(base_urgency)))
        else:
            urgency = "Low"
            urgency_score = min(38, max(5, round(base_urgency)))

    # Strict consistency enforcement between category and score
    urgency, urgency_score = validate_and_align_urgency(urgency, urgency_score)

    # Determine Sentiment Polarity
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "polarity_score": round(compound, 2),
        "urgency": urgency,
        "urgency_score": urgency_score,
    }


def _format_safe_gemini_error(exc: Exception) -> str:
    """Extracts safe diagnostic details from Gemini / HTTP exceptions without exposing secrets.

    Guarantees no request headers, auth tokens, or API keys are logged.
    """
    if isinstance(exc, APIError):
        status_code = getattr(exc, "code", None)
        status_str = getattr(exc, "status", None)
        msg = getattr(exc, "message", None) or str(exc)
        sanitized_msg = re.sub(
            r"(key|token|auth|secret)=[\w-]+",
            r"\1=[REDACTED]",
            str(msg),
            flags=re.IGNORECASE,
        )

        details = [f"type={type(exc).__name__}"]
        if status_code:
            details.append(f"http_status={status_code}")
        if status_str:
            details.append(f"api_status={status_str}")
        if sanitized_msg:
            details.append(f"message={sanitized_msg[:200]}")
        return ", ".join(details)

    if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
        return f"type={type(exc).__name__}, message=Request timed out after configured duration"

    if isinstance(exc, httpx.RequestError):
        return f"type={type(exc).__name__}, message=Network connection error ({type(exc).__name__})"

    if isinstance(exc, GeminiConfigurationError):
        return f"type={type(exc).__name__}, message={str(exc)}"

    return f"type={type(exc).__name__}, message={str(exc)[:150]}"


def analyze_text(text: str) -> dict:
    """Main complaint analysis entry point.

    Attempts primary analysis using Google Gemini API.
    If Gemini fails due to timeout, missing key, quota/rate limits, or network errors,
    it seamlessly falls back to the deterministic rule-based analyzer without exposing errors or secrets.
    """
    t0 = time.perf_counter()
    try:
        return _analyze_with_gemini(text)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        effective_timeout_ms = _get_effective_timeout()
        diag = _format_safe_gemini_error(exc)
        logger.warning(
            f"Gemini analysis unavailable after {elapsed:.2f}s "
            f"(effective_timeout_ms={effective_timeout_ms}, {diag}); "
            f"using rule-based fallback analyzer."
        )
        return fallback_analyze_text(text)
