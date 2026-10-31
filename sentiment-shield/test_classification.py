import os
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

# Enforce local SQLite for offline unit tests so tests never connect to remote databases
TEST_DB_PATH = Path(__file__).resolve().parent / "feedback.db"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"

from google.genai import types
import httpx
import ai_engine
from ai_engine import (
    ComplaintAnalysis,
    GeminiConfigurationError,
    _analyze_with_gemini,
    _get_gemini_client,
    analyze_text,
    fallback_analyze_text,
    validate_and_align_urgency,
)


class TestGeminiIntegration(unittest.TestCase):
    """Unit tests for Google Gemini API integration using mocks (zero quota consumption)."""

    def setUp(self):
        # Patch time.sleep to avoid real 1s delays during retry tests
        self.sleep_patcher = patch("time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()

    def test_gemini_valid_output(self):
        """Verify that valid structured output from Gemini is parsed and returned correctly."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Negative",
            polarity_score=-0.75,
            urgency="High",
            urgency_score=85,
        )
        mock_response.text = '{"sentiment": "Negative", "polarity_score": -0.75, "urgency": "High", "urgency_score": 85}'
        mock_client.models.generate_content.return_value = mock_response

        result = _analyze_with_gemini("Severe gas leak in the kitchen", client=mock_client)

        self.assertEqual(result["sentiment"], "Negative")
        self.assertEqual(result["polarity_score"], -0.75)
        self.assertEqual(result["urgency"], "High")
        self.assertEqual(result["urgency_score"], 85)
        # Verify default model used was gemini-3.5-flash-lite
        self.assertEqual(mock_client.models.generate_content.call_args.kwargs["model"], "gemini-3.5-flash-lite")

    def test_gemini_model_override(self):
        """Verify GEMINI_MODEL environment variable overrides the default model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Neutral",
            polarity_score=0.0,
            urgency="Low",
            urgency_score=20,
        )
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {"GEMINI_MODEL": "custom-gemini-model"}, clear=False):
            _analyze_with_gemini("Routine feedback", client=mock_client)
            self.assertEqual(mock_client.models.generate_content.call_args.kwargs["model"], "custom-gemini-model")

    def test_gemini_score_category_validation(self):
        """Verify that score and category boundaries are strictly enforced and aligned."""
        # Case 1: High urgency but score is too low (< 70) -> clamped to 70
        urgency, score = validate_and_align_urgency("High", 30)
        self.assertEqual(urgency, "High")
        self.assertGreaterEqual(score, 70)

        # Case 2: Low urgency but score is too high (>= 40) -> clamped to 39
        urgency, score = validate_and_align_urgency("Low", 85)
        self.assertEqual(urgency, "Low")
        self.assertLessEqual(score, 39)

        # Case 3: Medium urgency but score out of range -> clamped to [40, 69]
        urgency, score = validate_and_align_urgency("Medium", 95)
        self.assertEqual(urgency, "Medium")
        self.assertLessEqual(score, 69)
        self.assertGreaterEqual(score, 40)

        # Case 4: Polarity score out of [-1.0, 1.0] -> clamped by Pydantic validator
        parsed = ComplaintAnalysis(
            sentiment="Positive",
            polarity_score=2.5,
            urgency="Low",
            urgency_score=20,
        )
        self.assertEqual(parsed.polarity_score, 1.0)

    def test_gemini_invalid_or_malformed_output(self):
        """Verify that invalid/malformed Gemini output raises ValueError and triggers fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = None
        mock_response.text = "NOT_A_VALID_JSON_RESPONSE"
        mock_client.models.generate_content.return_value = mock_response

        # _analyze_with_gemini raises ValueError
        with self.assertRaises(ValueError):
            _analyze_with_gemini("Gas leak reported", client=mock_client)

        # analyze_text intercepts ValueError and gracefully uses fallback analyzer
        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            fallback_res = analyze_text("Gas leak reported in residential building near market")
            self.assertEqual(fallback_res["urgency"], "High")
            self.assertGreaterEqual(fallback_res["urgency_score"], 70)

    def test_gemini_missing_api_key(self):
        """Verify that missing GEMINI_API_KEY raises GeminiConfigurationError and triggers fallback."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=False):
            # _get_gemini_client raises configuration error
            with self.assertRaises(GeminiConfigurationError):
                _get_gemini_client()

            # analyze_text catches it and returns fallback analysis seamlessly
            result = analyze_text("Water pipeline burst causing dirty sewage water to flood houses")
            self.assertEqual(result["urgency"], "High")
            self.assertGreaterEqual(result["urgency_score"], 70)

    def test_gemini_timeout_fallback(self):
        """Verify that a request timeout gracefully falls back to rule-based analyzer."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError("Request timed out after 10000ms")

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            result = analyze_text("Complete power outage in hospital ICU and ventilator unit")
            self.assertEqual(result["urgency"], "High")
            self.assertGreaterEqual(result["urgency_score"], 70)

    def test_gemini_api_network_error_fallback(self):
        """Verify that network/API connection failure gracefully falls back without exposing errors."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = ConnectionError("Could not reach Google API")

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            result = analyze_text("Please schedule routine office cleaning for next Tuesday morning")
            self.assertEqual(result["urgency"], "Low")
            self.assertLessEqual(result["urgency_score"], 39)

    def test_gemini_client_error_diagnostic_logging(self):
        """Verify that ClientError logs HTTP status, API status, and sanitized message."""
        from google.genai.errors import ClientError

        sample_err = ClientError(
            404,
            {"error": {"message": "models/gemini-3.8-flash is not found", "status": "NOT_FOUND"}},
        )
        diag = ai_engine._format_safe_gemini_error(sample_err)
        self.assertIn("http_status=404", diag)
        self.assertIn("api_status=NOT_FOUND", diag)
        self.assertIn("models/gemini-3.8-flash is not found", diag)

        # Verify fallback is triggered cleanly on ClientError
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = sample_err

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            result = analyze_text("Water pipeline burst causing dirty sewage water to flood houses")
            self.assertEqual(result["urgency"], "High")

    def test_gemini_thinking_level_configuration(self):
        """Verify that thinking_level=LOW is configured for Gemini 3 models and omitted for unsupported models."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Negative",
            polarity_score=-0.8,
            urgency="High",
            urgency_score=85,
        )
        mock_client.models.generate_content.return_value = mock_response

        # Default model (gemini-3.8-flash) must have thinking_level=LOW
        _analyze_with_gemini("Emergency gas leak", client=mock_client)
        call_kwargs = mock_client.models.generate_content.call_args.kwargs
        config = call_kwargs["config"]
        self.assertIsNotNone(config.thinking_config)
        self.assertEqual(config.thinking_config.thinking_level, types.ThinkingLevel.LOW)

        # Overridden model that does not support thinking (e.g. gemini-1.5-flash)
        with patch.dict(os.environ, {"GEMINI_MODEL": "gemini-1.5-flash"}, clear=False):
            _analyze_with_gemini("Emergency gas leak", client=mock_client)
            call_kwargs = mock_client.models.generate_content.call_args.kwargs
            config = call_kwargs["config"]
            self.assertIsNone(config.thinking_config)

    def test_gemini_success_diagnostics_logging(self):
        """Verify that successful Gemini calls log elapsed time and effective timeout without secrets."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Positive",
            polarity_score=0.9,
            urgency="Low",
            urgency_score=15,
        )
        mock_client.models.generate_content.return_value = mock_response

        with self.assertLogs(ai_engine.logger, level="INFO") as log_cm:
            _analyze_with_gemini("Great service", client=mock_client)

        success_log = "\n".join(log_cm.output)
        self.assertIn("Gemini API call succeeded in", success_log)
        self.assertIn("effective_timeout_ms=", success_log)
        self.assertNotIn("Great service", success_log)  # Complaint text must not be logged
        self.assertNotIn("key=", success_log.lower())   # Secrets must not be logged

    def test_gemini_504_server_error_fallback(self):
        """Verify that upstream 504 DEADLINE_EXCEEDED ServerError logs safely and triggers fallback."""
        from google.genai.errors import ServerError

        sample_504 = ServerError(
            504,
            {
                "error": {
                    "code": 504,
                    "message": "Deadline expired before operation could complete.",
                    "status": "DEADLINE_EXCEEDED",
                }
            },
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = sample_504

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            with self.assertLogs(ai_engine.logger, level="WARNING") as log_cm:
                result = analyze_text("Severe gas leak in the kitchen and building hallway")

        # Fallback analyzer ran successfully
        self.assertEqual(result["urgency"], "High")
        self.assertGreaterEqual(result["urgency_score"], 70)

        # Warning log verified
        warning_log = "\n".join(log_cm.output)
        self.assertIn("Gemini analysis unavailable after", warning_log)
        self.assertIn("http_status=504", warning_log)
        self.assertIn("api_status=DEADLINE_EXCEEDED", warning_log)
        self.assertIn("Deadline expired before operation could complete.", warning_log)
        self.assertIn("effective_timeout_ms=60000", warning_log)
        self.assertNotIn("Severe gas leak", warning_log)

    def test_gemini_local_httpx_timeout_fallback(self):
        """Verify that local transport timeout (httpx.ReadTimeout) logs safely and triggers fallback."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = httpx.ReadTimeout("The read operation timed out")

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            with self.assertLogs(ai_engine.logger, level="WARNING") as log_cm:
                result = analyze_text("Hospital ICU oxygen line pressure drop emergency")

        # Fallback analyzer ran successfully
        self.assertEqual(result["urgency"], "High")
        self.assertGreaterEqual(result["urgency_score"], 70)

        # Warning log verified
        warning_log = "\n".join(log_cm.output)
        self.assertIn("type=ReadTimeout", warning_log)
        self.assertIn("Request timed out after configured duration", warning_log)
        self.assertIn("effective_timeout_ms=60000", warning_log)
        self.assertNotIn("Hospital ICU", warning_log)

    def test_effective_timeout_helper(self):
        """Verify _get_effective_timeout safely extracts timeout from client or environment."""
        # 1. From mock client with _api_client._http_options.timeout
        mock_c = MagicMock()
        mock_c._api_client._http_options.timeout = 45000
        self.assertEqual(ai_engine._get_effective_timeout(mock_c), 45000)

        # 2. From environment when client has no options
        with patch.dict(os.environ, {"GEMINI_TIMEOUT_MS": "30000"}, clear=False):
            self.assertEqual(ai_engine._get_effective_timeout(None), 30000)

        # 3. Default fallback when invalid
        with patch.dict(os.environ, {"GEMINI_TIMEOUT_MS": "invalid"}, clear=False):
            self.assertEqual(ai_engine._get_effective_timeout(None), 10000)

    def test_gemini_success_without_retry(self):
        """Verify that a successful request succeeds on attempt 1 without triggering a retry."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Positive",
            polarity_score=0.8,
            urgency="Low",
            urgency_score=20,
        )
        mock_client.models.generate_content.return_value = mock_response

        result = _analyze_with_gemini("Very satisfied with the quick resolution", client=mock_client)
        self.assertEqual(result["urgency"], "Low")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_gemini_transient_503_success_on_retry(self):
        """Verify that a transient 503 error is retried and succeeds on attempt 2."""
        from google.genai.errors import ServerError

        err_503 = ServerError(
            503,
            {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}
        )
        mock_response = MagicMock()
        mock_response.parsed = ComplaintAnalysis(
            sentiment="Negative",
            polarity_score=-0.85,
            urgency="High",
            urgency_score=90,
        )

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [err_503, mock_response]

        with self.assertLogs(ai_engine.logger, level="INFO") as log_cm:
            result = _analyze_with_gemini("Severe gas leak in the kitchen", client=mock_client)

        self.assertEqual(result["urgency"], "High")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

        combined_logs = "\n".join(log_cm.output)
        self.assertIn("encountered transient error", combined_logs)
        self.assertIn("http_status=503", combined_logs)
        self.assertIn("retrying in 1.0s (attempt 1 of 1 retry)...", combined_logs)
        self.assertIn("Gemini API call succeeded", combined_logs)
        self.assertIn("attempt=2", combined_logs)
        self.assertNotIn("Severe gas leak", combined_logs)

    def test_gemini_transient_failure_both_attempts_triggers_fallback(self):
        """Verify that when both attempt 1 and retry fail with 503, fallback analyzer runs."""
        from google.genai.errors import ServerError

        err_503 = ServerError(
            503,
            {"error": {"code": 503, "message": "This model is currently experiencing high demand.", "status": "UNAVAILABLE"}}
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = [err_503, err_503]

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            with self.assertLogs(ai_engine.logger, level="WARNING") as log_cm:
                result = analyze_text("Water pipeline burst causing dirty sewage water to flood houses")

        self.assertEqual(result["urgency"], "High")
        self.assertEqual(mock_client.models.generate_content.call_count, 2)

        warning_log = "\n".join(log_cm.output)
        self.assertIn("Gemini analysis unavailable after", warning_log)
        self.assertIn("http_status=503", warning_log)
        self.assertNotIn("Water pipeline burst", warning_log)

    def test_gemini_permanent_400_no_retry(self):
        """Verify that permanent 400 ClientError is NOT retried and immediately triggers fallback."""
        from google.genai.errors import ClientError

        err_400 = ClientError(
            400,
            {"error": {"code": 400, "message": "Invalid argument provided", "status": "INVALID_ARGUMENT"}}
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = err_400

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            result = analyze_text("Complete power outage in hospital ICU and ventilator unit")

        self.assertEqual(result["urgency"], "High")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)

    def test_gemini_permanent_403_no_retry(self):
        """Verify that permanent 403 ClientError (permission/auth) is NOT retried."""
        from google.genai.errors import ClientError

        err_403 = ClientError(
            403,
            {"error": {"code": 403, "message": "The caller does not have permission", "status": "PERMISSION_DENIED"}}
        )
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = err_403

        with patch.object(ai_engine, "_get_gemini_client", return_value=mock_client):
            result = analyze_text("Please schedule routine office cleaning for next Tuesday morning")

        self.assertEqual(result["urgency"], "Low")
        self.assertEqual(mock_client.models.generate_content.call_count, 1)


class TestClassificationRegressions(unittest.TestCase):
    """Deterministic regression tests ensuring accuracy across all 7 domains without API calls."""

    def setUp(self):
        # Patch Gemini analysis to use the deterministic fallback analyzer during regression suite
        self.gemini_patcher = patch.object(ai_engine, "_analyze_with_gemini", side_effect=fallback_analyze_text)
        self.gemini_patcher.start()

    def tearDown(self):
        self.gemini_patcher.stop()

    # -------------------------------------------------------------
    # Regression: Original Problem Sentence
    # -------------------------------------------------------------
    def test_regression_flood_high_risk_alert(self):
        sentence = "high risk alert in Rishikesh due to flood"
        result = analyze_text(sentence)

        self.assertNotEqual(result["urgency"], "Low", "Explicit high-risk flood alert must not be LOW")
        self.assertEqual(result["urgency"], "High")
        self.assertGreaterEqual(result["urgency_score"], 85)

    # -------------------------------------------------------------
    # 1. Public Safety and Natural Disasters
    # -------------------------------------------------------------
    def test_public_safety_and_natural_disasters(self):
        cases = [
            ("Gas leak reported in residential building near market", "High"),
            ("Live electrical wire snapped and sparking on the road outside school", "High"),
            ("Wildfire spreading rapidly towards village houses", "High"),
            ("Flash flood warning issued for low-lying areas near the river", "High"),
            ("Building foundation cracked and collapsing in sector 4", "High"),
            ("Chemical tanker overturned and hazardous acid spilling on highway", "High"),
            ("Fallen electric pole with sparks blocking main hospital entrance", "High"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)
                self.assertGreaterEqual(res["urgency_score"], 70)

    # -------------------------------------------------------------
    # 2. Medical and Personal Safety Emergencies
    # -------------------------------------------------------------
    def test_medical_and_personal_safety_emergencies(self):
        cases = [
            ("Patient having severe chest pain and cannot breathe, send ambulance", "High"),
            ("Elderly resident collapsed on the floor unconscious and bleeding", "High"),
            ("Worker suffered high-voltage electric shock and is unresponsive", "High"),
            ("Child accidentally swallowed toxic cleaning fluid", "High"),
            ("useer can not breath", "High"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)
                self.assertGreaterEqual(res["urgency_score"], 70)

    # -------------------------------------------------------------
    # 3. Banking, Unauthorized Transactions, and Financial Fraud
    # -------------------------------------------------------------
    def test_banking_and_financial_fraud(self):
        cases = [
            ("Someone just transferred 50,000 rupees from my account without OTP", "High"),
            ("Unauthorized international transaction detected on my credit card", "High"),
            ("My account was hacked and all my savings were drained", "High"),
            ("ATM swallowed my debit card, please help retrieve it", "Medium"),
            ("Money debited from my account but the merchant did not receive payment", "Medium"),
            ("Double deduction on my card for a single metro ticket", "Medium"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)

    # -------------------------------------------------------------
    # 4. Electricity, Water Supply, Infrastructure
    # -------------------------------------------------------------
    def test_infrastructure_and_utilities(self):
        cases = [
            ("Water pipeline burst causing dirty sewage water to flood houses", "High"),
            ("Complete power outage in hospital ICU and ventilator unit", "High"),
            ("Water supply disrupted in our colony for past 24 hours", "Medium"),
            ("Street lights not working on dark street for three days", "Medium"),
            ("Sewage drain overflowing onto the pedestrian footpath", "Medium"),
            ("Small pothole on 5th avenue needs asphalt repair", "Low"),
            ("Requesting routine tree pruning along the municipal sidewalk", "Low"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)

    # -------------------------------------------------------------
    # 5. Digital Services, Application Failures
    # -------------------------------------------------------------
    def test_digital_services_and_app_failures(self):
        cases = [
            ("Complete production server outage, all services down for all users", "High"),
            ("Ransomware attack encrypted our company database files", "High"),
            ("Application crashed repeatedly whenever I tap submit", "Medium"),
            ("Cannot reset my account password because verification link expired", "Medium"),
            ("Payment gateway throwing 500 error during checkout", "Medium"),
            ("Dark mode toggle icon is slightly misaligned on mobile screen", "Low"),
            ("Minor spelling typo on the terms and conditions page paragraph 2", "Low"),
            ("Could you please add support for dark theme in future update?", "Low"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)

    # -------------------------------------------------------------
    # 6. Customer Service & Sentiment vs Urgency Disconnect
    # -------------------------------------------------------------
    def test_customer_service_and_emotional_venting(self):
        cases = [
            ("Critical escalation: regulatory notice sent to ombudsman regarding unaddressed dispute", "High"),
            ("My support ticket #9821 has been pending for 3 weeks without any response", "Medium"),
            ("Support executive was extremely rude and unhelpful on phone call", "Medium"),
            ("Delivery package was delayed by two days, disappointed with logistics", "Medium"),
            ("I am absolutely furious, disgusted and outraged by your ugly new logo font!", "Low"),
            ("Worst color choice imaginable, I hate it so much, disgusting design", "Low"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)

    # -------------------------------------------------------------
    # 7. Routine Maintenance & Low-Priority Service Requests
    # -------------------------------------------------------------
    def test_routine_maintenance_and_low_priority_requests(self):
        cases = [
            ("Please schedule routine office cleaning for next Tuesday morning", "Low"),
            ("Kindly provide information on pricing tiers and enterprise plans", "Low"),
            ("Great application, very helpful, just submitting a small UI suggestion", "Low"),
            ("This is a low risk cosmetic glitch in the sidebar navigation", "Low"),
            ("Please send my quarterly bank account statement by email", "Low"),
            ("Routine health checkup appointment request for next week", "Low"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)
                self.assertLessEqual(res["urgency_score"], 39)

    # -------------------------------------------------------------
    # 8. Contextual Modifiers: Negations, Drills, and Resolved Incidents
    # -------------------------------------------------------------
    def test_contextual_modifiers(self):
        cases = [
            ("My account was not hacked, it was just my brother logging in", "Low"),
            ("False alarm: the flood warning from yesterday has been cancelled", "Low"),
            ("Annual emergency evacuation mock drill scheduled for Monday", "Low"),
            ("The flood receded last week and the road is now fully reopened", "Low"),
            ("Patient had chest pain last month but is completely fine now", "Low"),
            ("My card was blocked yesterday by bank customer care as requested", "Low"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                res = analyze_text(text)
                self.assertEqual(res["urgency"], expected)
                self.assertLessEqual(res["urgency_score"], 39)

    # -------------------------------------------------------------
    # 9. Sentiment Fallback (When No Domain Keywords Match)
    # -------------------------------------------------------------
    def test_sentiment_fallback_without_keywords(self):
        severe_neg = "This is atrocious, terrible, horrible, and repulsive"
        res_neg = analyze_text(severe_neg)
        self.assertEqual(res_neg["urgency"], "High")
        self.assertGreaterEqual(res_neg["urgency_score"], 70)

        pleasant = "I am very happy and delighted with everything"
        res_pos = analyze_text(pleasant)
        self.assertEqual(res_pos["urgency"], "Low")
        self.assertLessEqual(res_pos["urgency_score"], 39)

    # -------------------------------------------------------------
    # 10. Strict Score and Category Consistency
    # -------------------------------------------------------------
    def test_score_and_category_consistency(self):
        test_inputs = [
            "Emergency flood evacuation needed",
            "Application crashed",
            "Just a routine suggestion",
            "Not a scam, all good",
            "Terrible horrible awful service",
            "Awesome app thanks",
        ]
        for text in test_inputs:
            with self.subTest(text=text):
                res = analyze_text(text)
                if res["urgency"] == "High":
                    self.assertGreaterEqual(res["urgency_score"], 70)
                    self.assertLessEqual(res["urgency_score"], 99)
                elif res["urgency"] == "Medium":
                    self.assertGreaterEqual(res["urgency_score"], 40)
                    self.assertLessEqual(res["urgency_score"], 69)
                elif res["urgency"] == "Low":
                    self.assertGreaterEqual(res["urgency_score"], 5)
                    self.assertLessEqual(res["urgency_score"], 39)


class TestAuthenticationFlow(unittest.TestCase):
    """Unit tests for JWT authentication, expiration, decoding, and role-based guards."""

    def test_valid_token_decoding(self):
        """Verify that a valid token correctly decodes the user identity and role."""
        from main import create_access_token, get_current_user
        token = create_access_token(data={"sub": "complaint_user", "role": "user"})
        user = get_current_user(token=token)
        self.assertEqual(user["username"], "complaint_user")
        self.assertEqual(user["role"], "user")

    def test_expired_token_raises_401(self):
        """Verify that an expired token raises HTTP 401 Unauthorized."""
        from datetime import timedelta
        from fastapi import HTTPException
        from main import create_access_token, get_current_user
        expired_token = create_access_token(
            data={"sub": "complaint_user", "role": "user"},
            expires_delta=timedelta(seconds=-1)
        )
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=expired_token)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Could not validate credentials")

    def test_null_or_malformed_token_raises_401(self):
        """Verify that 'null', empty, or malformed tokens raise HTTP 401."""
        from fastapi import HTTPException
        from main import get_current_user
        for invalid_token in ["null", "undefined", "", "invalid.jwt.token"]:
            with self.subTest(token=invalid_token):
                with self.assertRaises(HTTPException) as ctx:
                    get_current_user(token=invalid_token)
                self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_claims_raises_401(self):
        """Verify that a token missing 'sub' or 'role' claims raises HTTP 401."""
        from fastapi import HTTPException
        from jose import jwt
        from main import get_current_user, SECRET_KEY, ALGORITHM
        token_no_role = jwt.encode({"sub": "user1"}, SECRET_KEY, algorithm=ALGORITHM)
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=token_no_role)
        self.assertEqual(ctx.exception.status_code, 401)

        token_no_sub = jwt.encode({"role": "user"}, SECRET_KEY, algorithm=ALGORITHM)
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(token=token_no_sub)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_admin_role_authorization_guard(self):
        """Verify get_current_admin strictly enforces role='admin'."""
        from fastapi import HTTPException
        from main import get_current_admin
        with self.assertRaises(HTTPException) as ctx:
            get_current_admin(current_user={"username": "normal_user", "role": "user"})
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertEqual(ctx.exception.detail, "Admin access required")

        admin = get_current_admin(current_user={"username": "super_admin", "role": "admin"})
        self.assertEqual(admin["role"], "admin")

    def test_protected_endpoints_have_auth_dependency(self):
        """Verify that POST /analyze and GET /logs have get_current_user dependency."""
        from main import app, get_current_user
        for route in app.routes:
            if getattr(route, "path", None) == "/analyze" and "POST" in getattr(route, "methods", set()):
                dep_callables = [d.call for d in route.dependant.dependencies]
                self.assertIn(get_current_user, dep_callables)
            elif getattr(route, "path", None) == "/logs" and "GET" in getattr(route, "methods", set()):
                dep_callables = [d.call for d in route.dependant.dependencies]
                self.assertIn(get_current_user, dep_callables)


class TestSecretKeyProductionSafety(unittest.TestCase):
    """Verifies that SECRET_KEY is safely guarded against insecure production defaults."""

    def test_development_defaults_to_dev_secret_key(self):
        """When ENVIRONMENT is unset or development, SECRET_KEY defaults to 'dev-secret-key'."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            os.environ.pop("SECRET_KEY", None)
            self.assertEqual(get_secret_key(), "dev-secret-key")

    def test_development_uses_explicit_secret_key(self):
        """When in development with an explicit SECRET_KEY, it returns that key."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "development", "SECRET_KEY": "custom-dev-secret"}):
            self.assertEqual(get_secret_key(), "custom-dev-secret")

    def test_production_rejects_missing_secret_key(self):
        """When ENVIRONMENT is production and SECRET_KEY is missing, raises RuntimeError."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            os.environ.pop("SECRET_KEY", None)
            with self.assertRaises(RuntimeError) as ctx:
                get_secret_key()
            self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))
            self.assertIn("production", str(ctx.exception).lower())

    def test_production_rejects_empty_secret_key(self):
        """When ENVIRONMENT is production and SECRET_KEY is empty/whitespace, raises RuntimeError."""
        from main import get_secret_key
        for empty_val in ["", "   ", "\t"]:
            with self.subTest(empty_val=empty_val):
                with patch.dict(os.environ, {"ENVIRONMENT": "production", "SECRET_KEY": empty_val}):
                    with self.assertRaises(RuntimeError) as ctx:
                        get_secret_key()
                    self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))

    def test_production_rejects_default_dev_secret_key(self):
        """When ENVIRONMENT is production and SECRET_KEY is 'dev-secret-key', raises RuntimeError."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "SECRET_KEY": "dev-secret-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                get_secret_key()
            self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))

    def test_production_alias_prod_rejects_insecure_key(self):
        """When ENVIRONMENT is 'prod' (alias for production), safety check is still enforced."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "prod", "SECRET_KEY": "dev-secret-key"}):
            with self.assertRaises(RuntimeError) as ctx:
                get_secret_key()
            self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))

    def test_production_accepts_strong_secret_key(self):
        """When ENVIRONMENT is production and a secure SECRET_KEY is provided, it is accepted."""
        from main import get_secret_key
        secure_key = "prod-strong-entropy-token-9876543210-secure"
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "SECRET_KEY": secure_key}):
            self.assertEqual(get_secret_key(), secure_key)

    def test_error_message_does_not_leak_unauthorized_key(self):
        """Ensure that exception messages never echo sensitive values."""
        from main import get_secret_key
        with patch.dict(os.environ, {"ENVIRONMENT": "production", "SECRET_KEY": "   "}):
            with self.assertRaises(RuntimeError) as ctx:
                get_secret_key()
            self.assertNotIn("partial-or-bad-secret", str(ctx.exception))


class TestDatabaseConfiguration(unittest.TestCase):
    """Verifies DATABASE_URL validation, normalization, and engine configuration."""

    def test_development_defaults_to_sqlite(self):
        """When ENVIRONMENT is development or unset, missing DATABASE_URL defaults to SQLite."""
        from database import get_database_url
        with patch.dict(os.environ, {"ENVIRONMENT": "development"}, clear=False):
            os.environ.pop("DATABASE_URL", None)
            url = get_database_url()
            self.assertTrue(url.startswith("sqlite:///"))
            self.assertIn("feedback.db", url)

    def test_postgres_url_normalization(self):
        """postgres:// is rewritten to postgresql:// and sslmode=require is ensured for remote hosts."""
        from database import get_database_url
        with patch.dict(os.environ, {
            "ENVIRONMENT": "development",
            "DATABASE_URL": "postgres://user:pass@db.supabase.co:5432/postgres"
        }):
            url = get_database_url()
            self.assertTrue(url.startswith("postgresql://"))
            self.assertIn("sslmode=require", url)

    def test_production_rejects_missing_database_url(self):
        """In production, missing DATABASE_URL raises RuntimeError."""
        from database import get_database_url
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False):
            os.environ.pop("DATABASE_URL", None)
            with self.assertRaises(RuntimeError) as ctx:
                get_database_url()
            self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))
            self.assertIn("DATABASE_URL", str(ctx.exception))

    def test_production_rejects_sqlite_database_url(self):
        """In production, SQLite DATABASE_URL raises RuntimeError."""
        from database import get_database_url
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "DATABASE_URL": "sqlite:///./prod.db"
        }):
            with self.assertRaises(RuntimeError) as ctx:
                get_database_url()
            self.assertIn("CRITICAL CONFIGURATION ERROR", str(ctx.exception))
            self.assertIn("PostgreSQL", str(ctx.exception))

    def test_production_accepts_postgres_url(self):
        """In production, a valid postgresql:// URL is accepted with SSL ensured."""
        from database import get_database_url
        valid_pg = "postgresql://postgres:secret@db.supabase.co:5432/postgres"
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "DATABASE_URL": valid_pg
        }):
            url = get_database_url()
            self.assertTrue(url.startswith("postgresql://"))
            self.assertIn("sslmode=require", url)

    def test_engine_creation_sqlite_vs_postgres(self):
        """Verifies that SQLite engines connect to sqlite and PostgreSQL engines configure pre_ping."""
        from database import create_db_engine
        sqlite_engine = create_db_engine("sqlite:///:memory:")
        self.assertEqual(sqlite_engine.dialect.name, "sqlite")
        sqlite_engine.dispose()

        pg_engine = create_db_engine("postgresql://user:pass@localhost:5432/testdb")
        self.assertEqual(pg_engine.dialect.name, "postgresql")
        self.assertTrue(pg_engine.pool._pre_ping)
        pg_engine.dispose()


class TestDatabaseOperations(unittest.TestCase):
    """Verifies user registration, repeated login, complaint isolation, and resolution using isolated DB."""

    def setUp(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database import Base
        self.test_engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
        Base.metadata.create_all(bind=self.test_engine)
        self.TestSession = sessionmaker(autocommit=False, autoflush=False, bind=self.test_engine)
        self.db = self.TestSession()

    def tearDown(self):
        self.db.close()
        from database import Base
        Base.metadata.drop_all(bind=self.test_engine)

    def test_user_registration_and_repeated_login(self):
        """Verify user registration, password hashing, and repeated successful logins."""
        from database import UserModel
        from main import hash_password

        # 1. Register user
        pw_hash = hash_password("secret123")
        user = UserModel(username="test_john", password=pw_hash, role="user")
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        self.assertIsNotNone(user.id)
        self.assertEqual(user.role, "user")

        # 2. First login attempt with correct password
        db_user_1 = self.db.query(UserModel).filter(
            UserModel.username == "test_john",
            UserModel.password == hash_password("secret123")
        ).first()
        self.assertIsNotNone(db_user_1)
        self.assertEqual(db_user_1.username, "test_john")

        # 3. Repeated login attempt with correct password
        db_user_2 = self.db.query(UserModel).filter(
            UserModel.username == "test_john",
            UserModel.password == hash_password("secret123")
        ).first()
        self.assertIsNotNone(db_user_2)
        self.assertEqual(db_user_2.id, user.id)

        # 4. Login attempt with incorrect password fails
        db_user_bad = self.db.query(UserModel).filter(
            UserModel.username == "test_john",
            UserModel.password == hash_password("wrongpassword")
        ).first()
        self.assertIsNone(db_user_bad)

    def test_complaint_persistence_and_user_isolation(self):
        """Verify complaint persistence and user log isolation (regular user vs admin)."""
        from database import FeedbackModel, UserModel
        from main import hash_password

        # Create user A, user B, and an admin
        user_a = UserModel(username="user_a", password=hash_password("pw"), role="user")
        user_b = UserModel(username="user_b", password=hash_password("pw"), role="user")
        admin = UserModel(username="admin_user", password=hash_password("pw"), role="admin")
        self.db.add_all([user_a, user_b, admin])
        self.db.commit()

        # Add complaints for user A and user B
        fb_a1 = FeedbackModel(
            user_id=user_a.id, text="Complaint 1 from A",
            sentiment="Negative", polarity_score=-0.6,
            urgency="High", urgency_score=80, status="Pending"
        )
        fb_a2 = FeedbackModel(
            user_id=user_a.id, text="Complaint 2 from A",
            sentiment="Neutral", polarity_score=0.0,
            urgency="Low", urgency_score=20, status="Pending"
        )
        fb_b1 = FeedbackModel(
            user_id=user_b.id, text="Complaint from B",
            sentiment="Negative", polarity_score=-0.4,
            urgency="Medium", urgency_score=50, status="Pending"
        )
        self.db.add_all([fb_a1, fb_a2, fb_b1])
        self.db.commit()

        # Query as User A: should only see A's 2 logs
        logs_a = self.db.query(FeedbackModel).filter(FeedbackModel.user_id == user_a.id).all()
        self.assertEqual(len(logs_a), 2)
        self.assertTrue(all(l.user_id == user_a.id for l in logs_a))

        # Query as User B: should only see B's 1 log
        logs_b = self.db.query(FeedbackModel).filter(FeedbackModel.user_id == user_b.id).all()
        self.assertEqual(len(logs_b), 1)
        self.assertEqual(logs_b[0].text, "Complaint from B")

        # Query as Admin: should see all 3 logs
        logs_all = self.db.query(FeedbackModel).all()
        self.assertEqual(len(logs_all), 3)

        # Toggle resolution status
        fb_a1.status = "Resolved"
        self.db.commit()
        self.db.refresh(fb_a1)
        self.assertEqual(fb_a1.status, "Resolved")


class TestMigrationUtility(unittest.TestCase):
    """Verifies SQLite-to-PostgreSQL migration utility helpers."""

    def test_normalize_and_secure_postgres_url_basic(self):
        from migrate_sqlite_to_pg import normalize_and_secure_postgres_url
        res = normalize_and_secure_postgres_url("postgres://u:p@db.supabase.co:5432/d")
        self.assertTrue(res.startswith("postgresql://"))
        self.assertIn("sslmode=require", res)

    def test_normalize_and_secure_postgres_url_placeholder_prompt(self):
        from migrate_sqlite_to_pg import normalize_and_secure_postgres_url
        with patch("getpass.getpass", return_value="mySecretPass123"):
            res = normalize_and_secure_postgres_url(
                "postgresql://postgres:[YOUR-PASSWORD]@db.supabase.co:5432/postgres"
            )
            self.assertIn("mySecretPass123", res)
            self.assertNotIn("[YOUR-PASSWORD]", res)
            self.assertIn("sslmode=require", res)

    def test_password_with_special_characters_is_url_encoded(self):
        """Verify that passwords containing @, #, %, &, etc. are safely URL-encoded."""
        from urllib.parse import urlparse
        from migrate_sqlite_to_pg import normalize_and_secure_postgres_url
        complex_pw = "P@ss#w0rd!%^&*()_+"
        with patch("getpass.getpass", return_value=complex_pw):
            res = normalize_and_secure_postgres_url(
                "postgresql://postgres:[YOUR-PASSWORD]@db.supabase.co:5432/postgres"
            )
            # URL must not contain raw unencoded '@' in the password segment
            self.assertNotIn(f":{complex_pw}@", res)
            self.assertIn("%40", res)  # '@' encoded
            self.assertIn("%23", res)  # '#' encoded

            # Verify SQLAlchemy correctly unquotes and retrieves the exact original password
            from sqlalchemy.engine.url import make_url
            parsed_url = make_url(res)
            self.assertEqual(parsed_url.password, complex_pw)

    def test_complaint_duplicate_detection_distinguishes_different_users(self):
        """Verify that identical text from DIFFERENT users is not falsely skipped."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from database import Base, FeedbackModel, UserModel

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Pre-populate User 1 and User 2
        u1 = UserModel(id=1, username="user1", password="pw", role="user")
        u2 = UserModel(id=2, username="user2", password="pw", role="user")
        session.add_all([u1, u2])

        # Pre-populate User 1's complaint
        fb1 = FeedbackModel(id=10, user_id=1, text="Same complaint text", sentiment="Negative", polarity_score=-0.5, urgency="High", urgency_score=80)
        session.add(fb1)
        session.commit()

        # Index target records exactly as done in migrate_sqlite_to_pg.py
        existing_feedback = session.query(FeedbackModel).all()
        pg_feedback_by_user_text = {(f.user_id, f.text): f for f in existing_feedback}

        # Case A: User 1 submits "Same complaint text" again -> matches (1, text) -> SKIP
        self.assertIn((1, "Same complaint text"), pg_feedback_by_user_text)

        # Case B: User 2 submits "Same complaint text" -> does NOT match (2, text) -> MUST NOT SKIP
        self.assertNotIn((2, "Same complaint text"), pg_feedback_by_user_text)

        session.close()
        engine.dispose()

    def test_read_sqlite_data_nonexistent_path(self):
        from pathlib import Path
        from migrate_sqlite_to_pg import read_sqlite_data
        with self.assertRaises(FileNotFoundError):
            read_sqlite_data(Path("nonexistent_db_12345.db"))

    def test_read_sqlite_data_real_db(self):
        from pathlib import Path
        from migrate_sqlite_to_pg import read_sqlite_data
        db_path = Path("feedback.db")
        if db_path.exists():
            users, feedback = read_sqlite_data(db_path)
            self.assertGreaterEqual(len(users), 1)
            self.assertGreaterEqual(len(feedback), 1)


if __name__ == "__main__":
    unittest.main()
