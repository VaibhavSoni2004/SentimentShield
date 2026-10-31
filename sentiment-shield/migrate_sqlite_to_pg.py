"""ComplaintFlow — SQLite to PostgreSQL Migration Utility

Transfers existing users and complaint feedback logs from a local SQLite database (feedback.db)
into a target PostgreSQL database (e.g., Supabase or Render PostgreSQL) safely and idempotently.

Safety guarantees:
1. STRICT READ-ONLY on SQLite: The source SQLite database is connected in URI read-only mode
   (?mode=ro) and is NEVER modified, overwritten, or deleted.
2. Safe Credential Handling & URL-Encoding: Passwords do not need to be written in terminal commands.
   The script prompts securely via getpass (or reads environment variables). Passwords with special
   characters (e.g., @, #, %, &, !) are automatically URL-encoded with urllib.parse.quote so connection
   strings parse safely without breaking host resolution.
3. Fully Idempotent Duplicate Detection: Identifies existing complaints strictly by (user_id, text).
   Re-running the migration never duplicates complaints, never overwrites existing records, and never
   skips complaints belonging to different users even if they have identical text.
4. User-Complaint Integrity: Maintains a dynamic ID map between SQLite and PostgreSQL users so
   complaints remain accurately linked to their author even if an admin was already created in PostgreSQL.
5. Sequence Synchronization: Automatically updates PostgreSQL serial sequences so new registrations
   and complaints receive unique, non-colliding IDs.
6. Dry-run mode: Allows verifying data without writing anything to PostgreSQL.

Usage:
    # Option A: Read TARGET_DATABASE_URL or DATABASE_URL from .env:
    python migrate_sqlite_to_pg.py

    # Option B: Prompt for password securely (keeps password out of terminal command history):
    python migrate_sqlite_to_pg.py --prompt-password

    # Option C: Pass URL with placeholder (script prompts for password):
    python migrate_sqlite_to_pg.py --target-url "postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres"

    # Option D: Dry-run simulation (inspect and verify without writing):
    python migrate_sqlite_to_pg.py --dry-run
"""

import argparse
import getpass
import logging
import os
import sqlite3
import sys
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse, urlunparse

from dotenv import dotenv_values, load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Locate paths
BASE_DIR = Path(__file__).resolve().parent
# Load .env with override=True so empty environment variables in shell sessions do not mask .env values
load_dotenv(BASE_DIR / ".env", override=True)
if (Path.cwd() / ".env").exists() and (Path.cwd() / ".env") != (BASE_DIR / ".env"):
    load_dotenv(Path.cwd() / ".env", override=True)

# Import models from database module
sys.path.insert(0, str(BASE_DIR))
from database import Base, FeedbackModel, UserModel

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("migration")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Safely migrate ComplaintFlow data from SQLite to PostgreSQL."
    )
    parser.add_argument(
        "--sqlite-path",
        type=str,
        default=str(BASE_DIR / "feedback.db"),
        help="Path to source SQLite feedback.db (default: sentiment-shield/feedback.db)",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        default="",
        help="Target PostgreSQL connection URL (defaults to TARGET_DATABASE_URL or DATABASE_URL env var)",
    )
    parser.add_argument(
        "--prompt-password",
        action="store_true",
        help="Securely prompt for the PostgreSQL password in the terminal (prevents storing password in shell history)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the migration without writing to PostgreSQL",
    )
    return parser.parse_args()


def normalize_and_secure_postgres_url(raw_url: str, force_prompt_password: bool = False) -> str:
    """Normalizes postgres:// -> postgresql://, enforces SSL for remote hosts,

    and securely prompts for and URL-encodes passwords with special characters.
    """
    url = raw_url.strip().strip("'\"") if raw_url else ""

    # Fall back to environment variables if not passed via CLI
    if not url:
        url = (os.getenv("TARGET_DATABASE_URL") or os.getenv("DATABASE_URL") or "").strip().strip("'\"")

    # If still not found, read directly from .env file using dotenv_values (bypasses any parent-shell masking)
    if not url:
        for candidate in [BASE_DIR / ".env", Path.cwd() / ".env"]:
            if candidate.exists():
                vals = dotenv_values(candidate)
                found = vals.get("TARGET_DATABASE_URL") or vals.get("DATABASE_URL")
                if found and found.strip():
                    url = found.strip().strip("'\"")
                    break

    if not url:
        raise ValueError(
            f"No target database URL provided. Checked command-line arguments, environment variables, "
            f"and '{BASE_DIR / '.env'}'. Please ensure DATABASE_URL is saved in .env or pass --target-url."
        )

    # Normalize 'postgres://' to 'postgresql://' for SQLAlchemy 1.4+ compatibility
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    # Detect placeholders before urlparse to prevent Python 3.13 bracketed IPv6 parser error
    placeholders = ["[YOUR-PASSWORD]", "[PASSWORD]", "<password>", "YOUR_PASSWORD"]
    has_placeholder = any(p in url for p in placeholders)

    if has_placeholder:
        pw = getpass.getpass("Enter PostgreSQL database password: ")
        if not pw:
            raise ValueError("Password cannot be empty.")
        # URL-encode password so special characters (@, #, %, &, etc.) do not break connection strings
        encoded_pw = quote(pw, safe="")
        for p in placeholders:
            url = url.replace(p, encoded_pw)

    parsed = urlparse(url)

    if force_prompt_password or (parsed.username and not parsed.password):
        user_label = parsed.username or "postgres"
        pw = getpass.getpass(f"Enter PostgreSQL password for user '{user_label}': ")
        if not pw:
            raise ValueError("Password cannot be empty.")

        encoded_pw = quote(pw, safe="")
        netloc = f"{user_label}:{encoded_pw}@{parsed.hostname}"
        if parsed.port:
            netloc += f":{parsed.port}"
        url = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))
        parsed = urlparse(url)

    # Re-parse to enforce SSL for remote PostgreSQL hosts (e.g. Supabase, Render)
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
        qs = parse_qs(parsed.query)
        if "sslmode" not in qs:
            delimiter = "&" if parsed.query else "?"
            url = f"{url}{delimiter}sslmode=require"

    return url


def read_sqlite_data(sqlite_path: Path):
    """Reads users and feedback logs from SQLite in strict read-only mode."""
    if not sqlite_path.exists():
        raise FileNotFoundError(f"SQLite database not found at: {sqlite_path}")

    # Use as_uri() + ?mode=ro to guarantee the engine rejects any write operations
    uri_path = f"{sqlite_path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri_path, uri=True)
    except sqlite3.OperationalError:
        # Fallback if URI mode is unsupported on older OS builds
        conn = sqlite3.connect(str(sqlite_path.resolve()))

    cursor = conn.cursor()

    # Verify tables exist
    existing_tables = [
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    ]

    users = []
    if "users" in existing_tables:
        users = cursor.execute(
            "SELECT id, username, password, role FROM users ORDER BY id ASC;"
        ).fetchall()

    feedback_logs = []
    if "feedback_logs" in existing_tables:
        feedback_logs = cursor.execute(
            "SELECT id, user_id, text, sentiment, polarity_score, urgency, urgency_score, status "
            "FROM feedback_logs ORDER BY id ASC;"
        ).fetchall()

    conn.close()
    return users, feedback_logs


def migrate_data(
    sqlite_path_str: str,
    target_url: str = "",
    prompt_password: bool = False,
    dry_run: bool = False,
):
    sqlite_path = Path(sqlite_path_str).resolve()
    logger.info(f"Connecting to source SQLite database (READ-ONLY): {sqlite_path}")

    users, feedback_logs = read_sqlite_data(sqlite_path)
    logger.info(f"Found {len(users)} user(s) and {len(feedback_logs)} feedback record(s) in SQLite.")

    try:
        normalized_url = normalize_and_secure_postgres_url(
            target_url, force_prompt_password=prompt_password
        )
    except ValueError as val_err:
        logger.error(str(val_err))
        sys.exit(1)

    if normalized_url.startswith("sqlite"):
        logger.error(
            "Target URL is SQLite. The migration target must be a PostgreSQL database (e.g. Supabase)."
        )
        sys.exit(1)

    if dry_run:
        logger.info("[DRY RUN MODE] Simulating migration. Zero changes will be written to target.")

    # Create engine for PostgreSQL with SSL and connection health checks
    connect_args = {}
    parsed = urlparse(normalized_url)
    if parsed.hostname and parsed.hostname not in ("localhost", "127.0.0.1"):
        connect_args["sslmode"] = "require"

    pg_engine = create_engine(
        normalized_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    try:
        # Verify connection without exposing credentials
        with pg_engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        masked_url = pg_engine.url.render_as_string(hide_password=True)
        logger.info(f"Successfully connected to target PostgreSQL: {masked_url}")
    except Exception as exc:
        logger.error(f"Could not connect to target PostgreSQL database: {exc}")
        sys.exit(1)

    from sqlalchemy import inspect
    inspector = inspect(pg_engine)
    has_users_table = inspector.has_table("users")
    has_feedback_table = inspector.has_table("feedback_logs")

    if not dry_run:
        # Ensure schema exists in PostgreSQL
        if not has_users_table or not has_feedback_table:
            logger.info("Creating missing tables (users, feedback_logs) in target PostgreSQL...")
            Base.metadata.create_all(bind=pg_engine)
            inspector = inspect(pg_engine)
            has_users_table = True
            has_feedback_table = True

    Session = sessionmaker(bind=pg_engine)
    session = Session()

    users_inserted = 0
    users_skipped = 0
    feedback_inserted = 0
    feedback_skipped = 0

    # Maps SQLite user_id -> PostgreSQL user_id to maintain strict relationship integrity
    user_id_map: dict[int, int] = {}

    try:
        # ---------------------------------------------------------
        # 1. Migrate Users
        # ---------------------------------------------------------
        logger.info("--- Migrating Users ---")
        existing_users = session.query(UserModel).all() if has_users_table else []
        pg_users_by_username = {u.username: u for u in existing_users}
        pg_users_by_id = {u.id: u for u in existing_users}

        for u in users:
            u_id, username, password, role = u

            if username in pg_users_by_username:
                # User already exists in PostgreSQL (e.g. created by create_default_admin)
                target_user = pg_users_by_username[username]
                user_id_map[u_id] = target_user.id
                users_skipped += 1
                logger.info(
                    f"  [EXISTS] User '{username}' already exists in PostgreSQL (id={target_user.id}). "
                    f"Preserved existing account and mapped complaints to this user."
                )
            elif u_id in pg_users_by_id:
                # ID collision with a different username in PostgreSQL: insert with next auto-assigned ID
                users_inserted += 1
                if not dry_run:
                    new_user = UserModel(
                        username=username,
                        password=password,
                        role=role or "user",
                    )
                    session.add(new_user)
                    session.flush()
                    user_id_map[u_id] = new_user.id
                    pg_users_by_username[username] = new_user
                    pg_users_by_id[new_user.id] = new_user
                else:
                    user_id_map[u_id] = u_id
                logger.info(
                    f"  [ADD]    User '{username}' inserted with auto-assigned ID (avoided ID collision with existing user)."
                )
            else:
                # Clean insert with preserved SQLite ID
                users_inserted += 1
                user_id_map[u_id] = u_id
                if not dry_run:
                    new_user = UserModel(
                        id=u_id,
                        username=username,
                        password=password,
                        role=role or "user",
                    )
                    session.add(new_user)
                    session.flush()
                    pg_users_by_username[username] = new_user
                    pg_users_by_id[u_id] = new_user
                logger.info(f"  [ADD]    User id={u_id} ('{username}', role={role})")

        if not dry_run:
            session.commit()

        # ---------------------------------------------------------
        # 2. Migrate Feedback Logs (Ordered after users)
        # ---------------------------------------------------------
        logger.info("--- Migrating Feedback Logs ---")
        existing_feedback = session.query(FeedbackModel).all() if has_feedback_table else []
        # Track both by primary key ID and by (user_id, text) pair for accurate duplicate detection
        pg_feedback_by_id = {f.id: f for f in existing_feedback}
        pg_feedback_by_user_text = {(f.user_id, f.text): f for f in existing_feedback}

        for f in feedback_logs:
            (
                f_id,
                sqlite_user_id,
                text_content,
                sentiment,
                polarity_score,
                urgency,
                urgency_score,
                status,
            ) = f

            # Remap to the verified PostgreSQL user ID
            target_user_id = user_id_map.get(sqlite_user_id, sqlite_user_id)

            # Check 1: Does this exact complaint already exist for THIS user in PostgreSQL?
            if (target_user_id, text_content) in pg_feedback_by_user_text:
                feedback_skipped += 1
                continue

            # Check 2: Does f_id already exist in PostgreSQL (occupied by a different complaint)?
            if f_id in pg_feedback_by_id:
                # ID collision: insert with auto-assigned ID so this user's complaint is not lost
                feedback_inserted += 1
                if not dry_run:
                    new_fb = FeedbackModel(
                        user_id=target_user_id,
                        text=text_content,
                        sentiment=sentiment,
                        polarity_score=polarity_score,
                        urgency=urgency,
                        urgency_score=urgency_score or 50,
                        status=status or "Pending",
                    )
                    session.add(new_fb)
                    session.flush()
                    pg_feedback_by_user_text[(target_user_id, text_content)] = new_fb
                logger.info(
                    f"  [ADD]    Complaint for user_id={target_user_id} inserted with auto-assigned ID (avoided collision with existing #{f_id})."
                )
            else:
                # Clean insert with preserved SQLite ID
                feedback_inserted += 1
                if not dry_run:
                    new_fb = FeedbackModel(
                        id=f_id,
                        user_id=target_user_id,
                        text=text_content,
                        sentiment=sentiment,
                        polarity_score=polarity_score,
                        urgency=urgency,
                        urgency_score=urgency_score or 50,
                        status=status or "Pending",
                    )
                    session.add(new_fb)
                    session.flush()
                    pg_feedback_by_id[f_id] = new_fb
                    pg_feedback_by_user_text[(target_user_id, text_content)] = new_fb

        if not dry_run:
            session.commit()
            logger.info(
                f"Feedback logs: {feedback_inserted} inserted, {feedback_skipped} already present."
            )

        # ---------------------------------------------------------
        # 3. Synchronize PostgreSQL Sequences (Prevents Future Collisions)
        # ---------------------------------------------------------
        if not dry_run and pg_engine.dialect.name == "postgresql":
            logger.info("--- Synchronizing PostgreSQL Sequences ---")
            with pg_engine.begin() as conn:
                for table_name in ["users", "feedback_logs"]:
                    try:
                        seq_res = conn.execute(
                            text(f"SELECT pg_get_serial_sequence('{table_name}', 'id')")
                        ).scalar()
                        if seq_res:
                            conn.execute(
                                text(
                                    f"SELECT setval('{seq_res}', "
                                    f"COALESCE((SELECT MAX(id) FROM {table_name}), 1), "
                                    f"(SELECT MAX(id) IS NOT NULL FROM {table_name}))"
                                )
                            )
                            logger.info(f"Sequence '{seq_res}' for table '{table_name}' synchronized.")
                    except Exception as seq_err:
                        logger.warning(
                            f"Sequence synchronization note for '{table_name}': {seq_err}"
                        )

        # Summary
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Source SQLite File:     {sqlite_path} (PRESERVED 100% UNTOUCHED)")
        print(f"Target Database:        {pg_engine.url.render_as_string(hide_password=True)}")
        print(f"Mode:                   {'DRY RUN (Preview Only)' if dry_run else 'LIVE MIGRATION'}")
        print(f"Users:                  {users_inserted} inserted, {users_skipped} already present")
        print(f"Feedback Logs:          {feedback_inserted} inserted, {feedback_skipped} already present")
        print("=" * 60)
        if not dry_run:
            print("Migration completed successfully! Original feedback.db is completely intact.\n")

    except Exception as exc:
        session.rollback()
        logger.error(f"Migration encountered an error and rolled back: {exc}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    args = parse_args()
    migrate_data(
        sqlite_path_str=args.sqlite_path,
        target_url=args.target_url,
        prompt_password=args.prompt_password,
        dry_run=args.dry_run,
    )
