"""
populate_online_courses.py
──────────────────────────
Reads the scraped JSON file and upserts every result row into the
`online_courses` table on MS SQL Server using Windows Integrated Security
(matching the Spring Boot connection setup).

Dependencies:
    pip install pyodbc

Usage:
    1. Edit the CONFIG section below to match your environment.
    2. python populate_online_courses.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit these values before running
# ═══════════════════════════════════════════════════════════════════════════════

# Path to your scraped JSON file
JSON_FILE = "scrapper_results.json"

# SQL Server connection settings (mirrors application.properties)
DB_SERVER   = "localhost"
DB_PORT     = 1433
DB_NAME     = "GP_db"

# How many rows to commit in one batch (tune for performance)
BATCH_SIZE  = 100

# Set to True to parse & print rows without touching the database
DRY_RUN     = False

# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pyodbc
except ImportError:
    sys.exit(
        "pyodbc is not installed.\n"
        "Run:  pip install pyodbc\n"
        "On Linux you may also need the ODBC Driver for SQL Server:\n"
        "  https://learn.microsoft.com/en-us/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server"
    )


# ─────────────────────────── helpers ────────────────────────────────────────

def detect_sql_server_driver() -> str:
    """
    Picks the best available MS SQL Server ODBC driver installed on this machine.
    Preference order: 18 → 17 → 13 → SQL Server (legacy).
    Exits with a helpful message if none are found.
    """
    preferred = [
        "ODBC Driver 18 for SQL Server",
        "ODBC Driver 17 for SQL Server",
        "ODBC Driver 13 for SQL Server",
        "SQL Server",                       # legacy driver, ships with Windows
    ]
    available = pyodbc.drivers()
    for driver in preferred:
        if driver in available:
            return driver

    # Nothing found — print what IS installed to help the user
    print("ERROR: No SQL Server ODBC driver found on this machine.")
    print("Installed ODBC drivers:")
    for d in available:
        print(f"  - {d}")
    print(
        "\nInstall the driver from:\n"
        "  https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server"
    )
    sys.exit(1)


def build_connection_string() -> str:
    """
    Builds a Windows-Auth (Integrated Security) connection string that mirrors
    the Spring Boot datasource URL in application.properties.
    Auto-detects the best available SQL Server ODBC driver.
    """
    driver = detect_sql_server_driver()
    print(f"Using ODBC driver: {driver}")
    return (
        f"DRIVER={{{driver}}};"
        f"SERVER={DB_SERVER},{DB_PORT};"
        f"DATABASE={DB_NAME};"
        f"Trusted_Connection=yes;"   # Windows Integrated Security
        f"Encrypt=no;"               # match encrypt=false in dev config
    )


def flatten_records(data: list[dict]) -> list[dict]:
    """
    Flattens the nested JSON structure.

    Input shape:
        [
          {
            "code": "AI311",
            "name": "Introduction to Logic",
            "results": [ { "id": ..., "source": ..., ... }, ... ]
          },
          ...
        ]

    Output: one flat dict per result row, enriched with course_code /
    course_name from the parent object.
    """
    rows = []
    for entry in data:
        course_code = entry.get("code", "")
        course_name = entry.get("name", "")
        for result in entry.get("results", []):
            rows.append(
                {
                    "id":          str(result.get("id", ""))[:500],
                    "course_code": str(course_code)[:30],
                    "course_name": str(course_name)[:300],
                    "source":      str(result.get("source", ""))[:50],
                    "title":       str(result.get("title", ""))[:300],
                    "url":         str(result.get("url", "") or "")[:1024],
                    "description": str(result.get("description", "") or "")[:2000],
                    "rating":      _to_float(result.get("rating")),
                    "reviews":     _to_int(result.get("reviews")),
                    "price":       _to_float(result.get("price")),
                    "score":       _to_float(result.get("score")),
                    "last_updated": datetime.now(),
                }
            )
    return rows


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ────────────────────────── SQL statements ──────────────────────────────────

CREATE_TABLE_SQL = """
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'online_courses'
)
BEGIN
    CREATE TABLE dbo.online_courses (
        id            NVARCHAR(500)  NOT NULL PRIMARY KEY,
        course_code   NVARCHAR(30)   NOT NULL,
        course_name   NVARCHAR(300)  NOT NULL,
        source        NVARCHAR(50)   NOT NULL,
        title         NVARCHAR(300)  NOT NULL,
        url           NVARCHAR(1024) NULL,
        description   NVARCHAR(2000) NULL,
        rating        FLOAT          NULL,
        reviews       INT            NULL,
        price         FLOAT          NULL,
        score         FLOAT          NULL,
        last_updated  DATETIME2      NULL
    );
    CREATE INDEX idx_online_courses_code ON dbo.online_courses (course_code);
    PRINT 'Table online_courses created.';
END
"""

# MERGE (upsert): update existing rows, insert new ones.
UPSERT_SQL = """
MERGE dbo.online_courses AS target
USING (SELECT ? AS id, ? AS course_code, ? AS course_name,
              ? AS source, ? AS title, ? AS url, ? AS description,
              ? AS rating, ? AS reviews, ? AS price, ? AS score,
              ? AS last_updated) AS src
ON target.id = src.id
WHEN MATCHED THEN
    UPDATE SET
        course_code  = src.course_code,
        course_name  = src.course_name,
        source       = src.source,
        title        = src.title,
        url          = src.url,
        description  = src.description,
        rating       = src.rating,
        reviews      = src.reviews,
        price        = src.price,
        score        = src.score,
        last_updated = src.last_updated
WHEN NOT MATCHED THEN
    INSERT (id, course_code, course_name, source, title, url,
            description, rating, reviews, price, score, last_updated)
    VALUES (src.id, src.course_code, src.course_name, src.source,
            src.title, src.url, src.description, src.rating, src.reviews,
            src.price, src.score, src.last_updated);
"""


# ─────────────────────────── main ───────────────────────────────────────────

def main():
    # ── 1. Load JSON ─────────────────────────────────────────────────────────
    json_path = Path(JSON_FILE)
    if not json_path.exists():
        sys.exit(f"File not found: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        sys.exit("Expected the JSON root to be a list of course objects.")

    rows = flatten_records(data)
    print(f"Parsed {len(rows)} result rows from {len(data)} course entries.")

    if DRY_RUN:
        print("Dry-run mode — first 3 rows:")
        for r in rows[:3]:
            print(" ", r)
        return

    # ── 2. Connect ───────────────────────────────────────────────────────────
    conn_str = build_connection_string()
    print(f"Connecting to {DB_SERVER},{DB_PORT}/{DB_NAME} …")
    try:
        conn = pyodbc.connect(conn_str, autocommit=False)
    except pyodbc.Error as exc:
        sys.exit(f"Connection failed:\n{exc}")

    cursor = conn.cursor()
    print("Connected.")

    # ── 3. Ensure table exists ────────────────────────────────────────────────
    cursor.execute(CREATE_TABLE_SQL)
    conn.commit()

    # ── 4. Upsert in batches ──────────────────────────────────────────────────
    inserted = updated = errors = 0

    for i, row in enumerate(rows, start=1):
        params = (
            row["id"],
            row["course_code"],
            row["course_name"],
            row["source"],
            row["title"],
            row["url"],
            row["description"],
            row["rating"],
            row["reviews"],
            row["price"],
            row["score"],
            row["last_updated"],
        )
        try:
            cursor.execute(UPSERT_SQL, params)
        except pyodbc.Error as exc:
            print(f"  [WARN] Row {i} skipped (id={row['id']!r}): {exc}")
            errors += 1
            continue

        # rowcount == 1 for both INSERT and UPDATE in a MERGE
        if cursor.rowcount == 1:
            inserted += 1          # rough count; MERGE doesn't distinguish here
        else:
            updated += 1

        if i % BATCH_SIZE == 0:
            conn.commit()
            print(f"  … committed {i}/{len(rows)} rows")

    conn.commit()   # final batch
    cursor.close()
    conn.close()

    print(
        f"\nDone. {inserted + updated} upserted "
        f"({errors} skipped due to errors) "
        f"out of {len(rows)} rows."
    )


if __name__ == "__main__":
    main()