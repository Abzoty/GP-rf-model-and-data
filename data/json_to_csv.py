import json
import csv
from collections import Counter

# ── File paths ─────────────────────────────────────────────────────────────────
STUDENTS_FILE            = 'students.json'
OUTPUT_FILE              = 'students_all.csv'
PRE_SPECIALIZATION_FILE  = 'before_specialization.json'  # used when filter is ON
# ──────────────────────────────────────────────────────────────────────────────

# ── Modular filters ────────────────────────────────────────────────────────────
# Set either flag to False (or comment out the block that applies it below)
# to disable that filter and revert to the full course set.

FILTER_TO_PRE_SPECIALIZATION = False   # Only include courses listed in PRE_SPECIALIZATION_FILE
FILTER_HU_COURSES            = False   # Exclude courses whose code starts with 'HU'
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_PROGRAMS = {
    "computer science",
    "operation research & decision support",
    "information systems",
    "information technology",
    "artificial intelligence",
}

# ── Load student data ──────────────────────────────────────────────────────────
with open(STUDENTS_FILE, 'r', encoding='utf-8') as f:
    students_data = json.load(f)

# ── Collect all course codes seen across target-program students ───────────────
course_codes = set()
for student in students_data:
    program = student.get('program') or ''
    program_norm = " ".join(program.split()).casefold()
    if program_norm in ALLOWED_PROGRAMS:
        for course in student.get('courses', []):
            code = course.get('code')
            if code:
                course_codes.add(code)

# ── Apply pre-specialization filter ───────────────────────────────────────────
# To disable: set FILTER_TO_PRE_SPECIALIZATION = False at the top, or
# comment out this entire block.
if FILTER_TO_PRE_SPECIALIZATION:
    with open(PRE_SPECIALIZATION_FILE, 'r', encoding='utf-8') as f:
        pre_spec_data = json.load(f)
    pre_spec_codes = {entry['code'] for entry in pre_spec_data if entry.get('code')}
    course_codes = course_codes & pre_spec_codes  # keep intersection only
# ──────────────────────────────────────────────────────────────────────────────

# ── Apply HU-course filter ─────────────────────────────────────────────────────
# To disable: set FILTER_HU_COURSES = False at the top, or
# comment out this entire block.
if FILTER_HU_COURSES:
    course_codes = {code for code in course_codes if not code.upper().startswith('HU')}
# ──────────────────────────────────────────────────────────────────────────────

course_codes = sorted(course_codes)

# ── Build CSV header ───────────────────────────────────────────────────────────
header = ['id']
for code in course_codes:
    header.extend([
        f'{code}_grade',
        f'{code}_points',
        f'{code}_termWork',
        f'{code}_examWork',
        f'{code}_result',
    ])
header.extend(['gpa', 'min_gpa', 'program'])

# ── Write CSV ──────────────────────────────────────────────────────────────────
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for student in students_data:
        student_id   = student.get('id')
        program      = student.get('program') or ''
        program_norm = " ".join(program.split()).casefold()

        if not program or program_norm == 'general' or program_norm not in ALLOWED_PROGRAMS:
            continue

        # Index this student's courses by code for fast lookup
        student_courses_dict = {}
        for course in student.get('courses', []):
            code = course.get('code')
            if code:
                student_courses_dict[code] = {
                    'grade':    course.get('grade',    ''),
                    'points':   course.get('points',   ''),
                    'termWork': course.get('termWork', ''),
                    'examWork': course.get('examWork', ''),
                    'result':   course.get('result',   ''),
                }

        row = [student_id]
        for code in course_codes:
            if code in student_courses_dict:
                c = student_courses_dict[code]
                row.extend([c['grade'], c['points'], c['termWork'], c['examWork'], c['result']])
            else:
                row.extend(['', '', '', '', ''])

        row.extend([student.get('gpa'), student.get('min_gpa'), program.strip()])
        writer.writerow(row)

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"Success! Data saved to '{OUTPUT_FILE}'")
print(f"Active filters: pre-specialization={FILTER_TO_PRE_SPECIALIZATION}, exclude-HU={FILTER_HU_COURSES}")
print(f"Unique course codes used : {len(course_codes)}")
print(f"Total columns created    : {len(header)}")

# Debug: unique programs in the file and their student counts
programs = sorted({
    str(student.get("program", "")).strip()
    for student in students_data
    if student.get("program")
})
print("\nPrograms found in JSON:")
for p in programs:
    print(f"  - {repr(p)}")

program_counts = Counter(
    str(student.get("program", "")).strip()
    for student in students_data
    if student.get("program")
)
print("\nProgram counts:")
for prog, count in sorted(program_counts.items()):
    print(f"  {prog}: {count}")