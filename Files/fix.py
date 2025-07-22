import os

def replace_in_file(filepath, old_string, new_string):
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()

    if old_string in content:
        content = content.replace(old_string, new_string)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated: {filepath}")

def replace_in_directory(directory, old_string, new_string):
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.py'):
                filepath = os.path.join(root, name)
                replace_in_file(filepath, old_string, new_string)

# === Call it like this ===
replace_in_directory('C:\\Users\\nicle\\Desktop\\Master-thesis-clean-code\\Files', 'ASNN', 'ASNN')
