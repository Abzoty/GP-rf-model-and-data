#!/usr/bin/env python3
"""
scrape_student_courses_refactored.py

Produces an anonymized JSON array of student objects and their courses.
"""
import os
import time
import json
import requests
from typing import Any, Dict, List, Optional

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIyMDIzMDYyMyIsImF1dGgiOiJST0xFX1NUVURFTlQiLCJleHAiOjE3ODIxNTEwOTF9.6KJrEQBNwAUfitwLa1yoZjombGwrFa8LTKSgnIf2dpErLn3shXAaHCEBiW7dQTO3U7vupxMXTL5FUV9g2ffUKg"      # Replace with your actual Bearer token
OUTPUT_PATH = r".\data\students.json" # Output JSON file path
DELAY = 0.05                          # Delay between requests (seconds)

URL_TEMPLATE = "http://newecom.fci-cu.edu.eg/api/student-courses?size=150&studentId.equals={student_id}&includeWithdraw.equals=true"

# Ranges to scrape
ID_RANGES = [
    (20230001, 20239999),
]
# ==========================================

def get_nested(obj: Dict[str, Any], path: str) -> Optional[Any]:
    """Safely get nested keys using dot notation. Returns None if any step missing."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def strip_if_str(val: Any) -> Any:
    """Helper to strip leading/trailing whitespace if the value is a string."""
    if isinstance(val, str):
        return val.strip()
    return val


def extract_course_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and order the fields requested from a course record, stripping strings."""
    return {
        "code": strip_if_str(get_nested(record, "course.code")),
        "name": strip_if_str(get_nested(record, "course.name")),
        "credit_hours": get_nested(record, "course.numOfHours"),
        "level": strip_if_str(get_nested(record, "course.level.name")),
        "term": strip_if_str(get_nested(record, "course.term.name")),
        "type": strip_if_str(get_nested(record, "course.type.name")),
        "grade": strip_if_str(record.get("grade")),
        "points": record.get("points"),
        "termWork": record.get("termWork"),
        "examWork": record.get("examWork"),
        "result": record.get("result"),
        "register_level": strip_if_str(get_nested(record, "level.name")),
        "register_term": strip_if_str(get_nested(record, "term.name")),
    }


def _unwrap_response_data(data: Any) -> List[Dict[str, Any]]:
    """Normalize possible API response shapes into a list of records."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "content" in data and isinstance(data["content"], list):
            return data["content"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    return []


def extract_student_meta_from_records(records: List[Dict[str, Any]]) -> Dict[str, Optional[Any]]:
    """Try to extract student-level metadata from the first valid record."""
    for rec in records:
        if not isinstance(rec, dict):
            continue
        student = rec.get("student")
        if isinstance(student, dict):
            return {
                "program": strip_if_str(get_nested(student, "program.name")),
                "min_gpa": get_nested(student, "program.minGpa"),
                "gpa": student.get("gpa"),
                "total": student.get("total"),
                "max_total": student.get("grade4TotalCourses"),
                "percentage": student.get("percentage"),
            }
            
    return {
        "program": None, "min_gpa": None, "gpa": None, 
        "total": None, "max_total": None, "percentage": None
    }


def fetch_for_student(student_id: int, session: requests.Session) -> Dict[str, Any]:
    """Fetch courses for a single student and return normalized structure."""
    url = URL_TEMPLATE.format(student_id=student_id)
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "User-Agent": "student-scraper/1.0",
    }

    try:
        resp = session.get(url, headers=headers, timeout=20)
    except requests.RequestException as e:
        return {"error": f"request-exception: {str(e)}", "courses": []}

    if resp.status_code != 200:
        return {"error": f"http-{resp.status_code}", "raw_text": resp.text[:1000], "courses": []}

    try:
        data = resp.json()
    except ValueError:
        return {"error": "invalid-json", "raw_text": resp.text[:1000], "courses": []}

    records = _unwrap_response_data(data)
    
    # Handle edge case where student has no records but data payload contains student object
    if not records:
        if isinstance(data, dict) and "student" in data and isinstance(data["student"], dict):
            student = data["student"]
            return {
                "program": strip_if_str(get_nested(student, "program.name")),
                "min_gpa": get_nested(student, "program.minGpa"),
                "gpa": student.get("gpa"),
                "total": student.get("total"),
                "max_total": student.get("grade4TotalCourses"),
                "percentage": student.get("percentage"),
                "courses": [],
            }
            
        return {"error": "unexpected-structure-or-empty", "courses": []}

    student_meta = extract_student_meta_from_records(records)
    extracted_courses = [extract_course_fields(rec) for rec in records]

    return {
        "program": student_meta.get("program"),
        "min_gpa": student_meta.get("min_gpa"),
        "gpa": student_meta.get("gpa"),
        "total": student_meta.get("total"),
        "max_total": student_meta.get("max_total"),
        "percentage": student_meta.get("percentage"),
        "courses": extracted_courses,
    }


def main():
    if not TOKEN or TOKEN == "YOUR_BEARER_TOKEN_HERE":
        print("ERROR: Please set a valid Bearer token in the script's configuration section.")
        return

    # Build the full list of student IDs from all ranges
    all_ids: List[int] = []
    for start, end in ID_RANGES:
        all_ids.extend(range(start, end + 1))

    total_to_do = len(all_ids)
    print(f"Will query {total_to_do} student IDs across {len(ID_RANGES)} ranges.")

    session = requests.Session()
    results: List[Dict[str, Any]] = []

    for idx, sid in enumerate(all_ids, start=1):
        item = fetch_for_student(sid, session)

        # Build final object in exact order requested, replacing standard student ID with script ID
        final_item: Dict[str, Any] = {
            "id": idx,
            "program": item.get("program"),
            "min_gpa": item.get("min_gpa"),
            "gpa": item.get("gpa"),
            "total": item.get("total"),
            "max_total": item.get("max_total"),
            "percentage": item.get("percentage"),
            "courses": item.get("courses", []),
        }

        # Include errors only for debugging if anything failed
        if "error" in item:
            final_item["error"] = item["error"]
            final_item["failed_original_studentId"] = sid

        results.append(final_item)

        if idx % 50 == 0:
            print(f"Progress: {idx}/{total_to_do} (last fetched ID index: {idx})")

        time.sleep(DELAY)

    # =============== NEW FIX ===============
    # Create the directory path if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:  # Checks if it's not an empty string
        os.makedirs(output_dir, exist_ok=True)
    # =======================================

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Done. Wrote {len(results)} items to '{OUTPUT_PATH}'")


if __name__ == "__main__":
    main()