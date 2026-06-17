"""
Script 1 — Extract unique courses from students enrolled in the 5 target departments.

For each unique course (identified by code) the script collects:
  code, name, arabicName, creditHours, type, level, term, courseDescription

NOTE: 'arabicName' and 'courseDescription' are not present in the current dataset;
      they are included as null to preserve the intended output schema.

Usage:
    python extract_unique_courses.py
"""

import json

# ── Global configuration — change these paths as needed ───────────────────────
INPUT_PATH  = "sample.json"          # Raw student data
OUTPUT_PATH = "unique_courses.json"  # Output: one object per unique course

# Target programs — matching is case-insensitive and whitespace-tolerant
TARGET_PROGRAMS = [
    "computer science",
    "operation research & decision support",
    "information systems",
    "information technology",
    "artificial intelligence",
]
# ──────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase and strip surrounding whitespace for reliable comparison."""
    return text.strip().lower() if text else ""


def extract_course_fields(course: dict) -> dict:
    """
    Pull the required fields from a raw course record.
    'credit_hours' (snake_case in source) is remapped to 'creditHours'.
    Fields absent from the dataset are kept as null for schema consistency.
    """
    return {
        "code":              course.get("code"),
        "name":              course.get("name"),
        "arabicName":        course.get("arabicName"),         # not in current data
        "creditHours":       course.get("credit_hours"),       # remapped from snake_case
        "type":              course.get("type"),
        "level":             course.get("level"),
        "term":              course.get("term"),
        "courseDescription": course.get("courseDescription"),  # not in current data
    }


def main():
    # Load the full student dataset
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        students = json.load(f)

    # Build a set of normalised target program names for O(1) lookup
    normalised_targets = {normalize(p) for p in TARGET_PROGRAMS}

    # Dict keyed by course code so duplicate encounters are silently skipped
    unique_courses: dict[str, dict] = {}

    for student in students:
        program = student.get("program", "")

        # Skip students whose program is not in the 5 target departments
        if normalize(program) not in normalised_targets:
            continue

        for course in student.get("courses", []):
            code = course.get("code")

            # Only record the first occurrence of each course code
            if code and code not in unique_courses:
                unique_courses[code] = extract_course_fields(course)

    # Sort by course code for deterministic, readable output
    result = sorted(unique_courses.values(), key=lambda c: c["code"] or "")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(
        f"[Script 1] Done — {len(result)} unique courses from "
        f"{sum(1 for s in students if normalize(s.get('program','')) in normalised_targets)} "
        f"target-department students  →  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
