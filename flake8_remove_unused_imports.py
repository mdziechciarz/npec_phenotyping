import os
import subprocess
import re

def remove_unused_imports(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    with open(file_path, 'w') as f:
        for line in lines:
            if not any(line.startswith(imp) for imp in unused_imports):
                f.write(line)

def main():
    result = subprocess.run(['flake8', '--select', 'F401', 'src/'], capture_output=True, text=True)
    output = result.stdout

    unused_imports = set()
    for line in output.splitlines():
        match = re.match(r'^.*(\S+)\s+(\S+)\s+.*$', line)
        if match:
            unused_imports.add(match.group(1))

    for root, _, files in os.walk('src/'):
        for file_name in files:
            if file_name.endswith('.py'):
                file_path = os.path.join(root, file_name)
                remove_unused_imports(file_path)

if __name__ == "__main__":
    main()
