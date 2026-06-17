import json

# Change these as needed
INPUT_JSON_PATH = "courses.json"
OUTPUT_JSON_PATH = "descriptions.json"

FIELDS_TO_KEEP = [
    "code",
    "name",
    "arabicName",
    "type",
    "courseDescription",
]

# Allowed course code prefixes
ALLOWED_PREFIXES = {
    "AI", "CS", "IT", "DS", "IS",   # main departments
    "MA", "HU", "ST", "TR",         # other essentials  
}


def pick_fields(course: dict) -> dict:
    """Return a new dict with only the selected fields."""
    return {field: course.get(field) for field in FIELDS_TO_KEEP}


def is_allowed_course(course: dict) -> bool:
    code = course.get("code", "")
    if not isinstance(code, str) or len(code) < 2:
        return False
    return code[:2].upper() in ALLOWED_PREFIXES


def main():
    with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
        courses_data = json.load(f)

    if not isinstance(courses_data, list):
        raise ValueError("Input JSON must be an array of course objects.")

    filtered_courses = [
        pick_fields(course)
        for course in courses_data
        if is_allowed_course(course)
    ]

    output_data = {
        "programs": [],
        "courses": filtered_courses,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(filtered_courses)} courses to {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()