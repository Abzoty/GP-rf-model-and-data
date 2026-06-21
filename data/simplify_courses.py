import json

# Define the input and output file names
input_file = 'courses_list.json'
output_file = 'courses_list.json'

# Define the prefixes we want to KEEP (using a tuple so .startswith() can use it directly)
ALLOWED_PREFIXES = ("CS", "IS", "IT", "AI", "DS", "MA", "HU", "ST", "TR")

def simplify_and_filter_courses():
    try:
        # Read the original JSON file
        with open(input_file, 'r', encoding='utf-8') as file:
            courses = json.load(file)
            
        filtered_courses = []
        
        for course in courses:
            code = course.get("code")
            
            # 1. Check if 'code' actually exists and is a string (prevents errors)
            # 2. Check if the code starts with any of the allowed prefixes
            if isinstance(code, str) and code.startswith(ALLOWED_PREFIXES):
                
                # If it passes the filter, keep only the code and name
                filtered_courses.append({
                    "code": code,
                    "name": course.get("name")
                })
        
        # Write the filtered data to a new JSON file
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(filtered_courses, file, indent=4)
            
        print(f"Success! Kept {len(filtered_courses)} courses out of {len(courses)}.")
        print(f"Data saved to '{output_file}'.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file name.")
    except json.JSONDecodeError:
        print("Error: The input file is not a valid JSON.")

if __name__ == "__main__":
    simplify_and_filter_courses()