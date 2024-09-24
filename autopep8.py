import os
import autopep8

def format_file(filepath):
    try:
        # Read the original content
        with open(filepath, 'r') as file:
            original_code = file.read()

        # Format the code with autopep8
        formatted_code = autopep8.fix_code(original_code, options={'max_line_length': 120})

        # Write back the formatted code
        with open(filepath, 'w') as file:
            file.write(formatted_code)

        print(f"Formatted: {filepath}")
    except Exception as e:
        print(f"Error formatting {filepath}: {str(e)}")

def format_src_folder():
    src_folder = 'src'  # Replace with your source folder path
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith('.py'):  # Process only Python files
                filepath = os.path.join(root, file)
                format_file(filepath)

if __name__ == "__main__":
    format_src_folder()
