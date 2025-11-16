"""Copy all project files from workspace to current directory."""
import os
import shutil
import json

# List of all files to copy (relative to tumor-evolution-clock/)
files_to_copy = [
    'README.md', 'ETHICS.md', 'requirements.txt', '.gitignore', 'LICENSE', 
    'QUICKSTART.md', 'test_pipeline.py',
    'scripts/simulate_clonal_trajectories.py',
    'scripts/__init__.py',
    'src/features.py', 'src/models.py', 'src/train.py', 'src/evaluate.py',
    'src/visualization.py', 'src/__init__.py',
    'notebooks/01_simulate_data.ipynb',
    'notebooks/02_clone_deconv.ipynb',
    'notebooks/03_modeling.ipynb'
]

# Try to find workspace path
workspace_paths = [
    'tumor-evolution-clock',
    '../tumor-evolution-clock',
    '../../tumor-evolution-clock'
]

workspace = None
for path in workspace_paths:
    if os.path.exists(path) and os.path.exists(os.path.join(path, 'README.md')):
        workspace = path
        break

if workspace:
    print(f"Found workspace at: {workspace}")
    for file_path in files_to_copy:
        src = os.path.join(workspace, file_path)
        dst = file_path
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else '.', exist_ok=True)
            shutil.copy2(src, dst)
            print(f"Copied: {file_path}")
        else:
            print(f"Not found: {file_path}")
else:
    print("Workspace not found. Files may need to be created manually.")
    print("Creating essential files...")
    
    # Create __init__.py files
    for dir_name in ['scripts', 'src']:
        init_file = os.path.join(dir_name, '__init__.py')
        os.makedirs(dir_name, exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f"# {dir_name} package\n")
            print(f"Created: {init_file}")

print("Done!")

