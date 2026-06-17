"""
Script 2 — Filter unique courses to those safe for model training.

Reads the output of Script 1 (unique_courses.json) and keeps only courses that:
  1. Belong to First Level or Second Level (Years 1–2, before specialization)
  2. Are NOT of type "Specialization Mandatory" (these leak department information)

Both conditions must be satisfied simultaneously (AND logic).

Usage:
    python filter_pre_specialization_courses.py
"""

import json

# ── Global configuration — change these paths as needed ───────────────────────
INPUT_PATH  = "courses.json"                 # Output from Script 1
OUTPUT_PATH = "pre_specialization_courses.json"     # Final leak-free course list

# Only courses at these levels are considered pre-specialisation
ALLOWED_LEVELS = {"first level", "second level"}

# This type reveals department identity and must be excluded entirely
EXCLUDED_TYPE = "specialization mandatory"
# ──────────────────────────────────────────────────────────────────────────────


def normalize(text: str) -> str:
    """Lowercase and strip surrounding whitespace for reliable comparison."""
    return text.strip().lower() if text else ""


def is_pre_specialization(course: dict) -> bool:
    """
    Return True only when BOTH conditions hold:
      - course level is Year 1 or Year 2
      - course type is not Specialization Mandatory
    """
    level_passes = normalize(course.get("level", "")) in ALLOWED_LEVELS
    type_passes  = normalize(course.get("type",  "")) != EXCLUDED_TYPE
    return level_passes and type_passes


def main():
    # Load the unique courses produced by Script 1
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        courses = json.load(f)

    # Apply both pre-specialisation filters
    filtered = [course for course in courses if is_pre_specialization(course)]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    removed = len(courses) - len(filtered)
    print(
        f"[Script 2] Done — kept {len(filtered)} / {len(courses)} courses "
        f"({removed} removed as post-specialisation or leaking)  →  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
