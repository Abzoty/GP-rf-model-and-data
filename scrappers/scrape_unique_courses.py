import os
import time
import json
import requests
from typing import Any, Dict, List, Optional

# ==========================================
# CONFIGURATION VARIABLES
# ==========================================
TOKEN = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiIyMDIzMDYyMyIsImF1dGgiOiJST0xFX1NUVURFTlQiLCJleHAiOjE3ODIxNTEwOTF9.6KJrEQBNwAUfitwLa1yoZjombGwrFa8LTKSgnIf2dpErLn3shXAaHCEBiW7dQTO3U7vupxMXTL5FUV9g2ffUKg"      # Replace with your actual Bearer token
OUTPUT_PATH = r".\data\courses.json"          # Output JSON file path (e.g., r"..\data\courses.json")
DELAY = 0.05                          # Delay between requests (seconds)
# ==========================================

URL_TEMPLATE = "http://newecom.fci-cu.edu.eg/api/student-courses?size=150&studentId.equals={student_id}&includeWithdraw.equals=true"

# Ranges to scrape — one range per enrolment year.
ID_RANGES = [
    (20180001, 20189999),
    (20190001, 20199999),
    (20200001, 20209999),
    (20210001, 20219999),
    (20220001, 20229999),
]


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


def extract_course(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract a Course entity from a student-course record.
    Returns None if the record has no course or no course code (can't deduplicate).
    """
    course = record.get("course")
    if not isinstance(course, dict):
        return None

    code = course.get("code")
    if not code:
        return None

    return {
        "code":              code,
        "name":              course.get("name"),
        "arabicName":        course.get("arabicName"),
        "creditHours":       course.get("numOfHours"),
        "type":              get_nested(course, "type.name"),
        "level":             get_nested(course, "level.name"),
        "term":              get_nested(course, "term.name"),
        "courseDescription": get_nested(course, "courseDescription.courseDescription"),
    }


def _unwrap_response_data(data: Any) -> List[Dict[str, Any]]:
    """Normalize possible API response shapes into a flat list of records."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "content" in data and isinstance(data["content"], list):
            return data["content"]
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
    return []


def fetch_courses_for_student(student_id: int, session: requests.Session) -> List[Dict[str, Any]]:
    """
    Fetch raw student-course records for a single student.
    Returns a (possibly empty) list of raw records, swallowing errors silently
    so the main loop can continue.
    """
    url = URL_TEMPLATE.format(student_id=student_id)
    headers = {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/json",
        "User-Agent": "course-scraper/1.0",
    }

    try:
        resp = session.get(url, headers=headers, timeout=20)
    except requests.RequestException:
        return []

    if resp.status_code != 200:
        return []

    try:
        data = resp.json()
    except ValueError:
        return []

    return _unwrap_response_data(data)


def main():
    if not TOKEN or TOKEN == "YOUR_BEARER_TOKEN_HERE":
        print("ERROR: Please set a valid Bearer token in the script's configuration section.")
        return

    delay = max(0.0, float(DELAY))

    # Build the full list of student IDs from all ranges
    all_ids: List[int] = []
    for start, end in ID_RANGES:
        all_ids.extend(range(start, end + 1))

    total_to_do = len(all_ids)
    print(f"Will query {total_to_do} student IDs across {len(ID_RANGES)} range(s):")
    for start, end in ID_RANGES:
        print(f"  {start} .. {end}  ({end - start + 1} IDs)")

    session = requests.Session()

    # Use course code as the deduplication key
    unique_courses: Dict[str, Dict[str, Any]] = {}

    for idx, sid in enumerate(all_ids, start=1):
        records = fetch_courses_for_student(sid, session)

        for record in records:
            course = extract_course(record)
            if not course:
                continue
            code = course["code"]
            if code not in unique_courses:
                # First time seeing this course — add it
                unique_courses[code] = course
            elif unique_courses[code]["term"] is None and course["term"] is not None:
                # Already stored but with a null term — patch it with the non-null value
                unique_courses[code]["term"] = course["term"]

        if idx % 50 == 0:
            print(f"Progress: {idx}/{total_to_do} | Unique courses found so far: {len(unique_courses)}")

        time.sleep(delay)

    # Sort by course code for a stable, readable output
    sorted_courses = sorted(unique_courses.values(), key=lambda c: c["code"])

    # Automatically create the output directory if it doesn't exist
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_courses, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Found {len(sorted_courses)} unique courses. Written to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    main()