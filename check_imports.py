"""
Validate codebase structure without requiring pip-installed packages.
Checks that all modules can be found and have no Python syntax errors.
"""
import sys
import ast
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

files_to_check = [
    "src/data/imagenet_loader.py",
    "src/experiments/probing.py",
    "src/analysis/cka.py",
    "src/analysis/visualization.py",
    "run_probing.py",
    "configs/probing_config.yaml",
]

print("Checking file existence and syntax...\n")
all_ok = True

for rel_path in files_to_check:
    full_path = os.path.join(ROOT, rel_path)
    if not os.path.exists(full_path):
        print(f"  [MISSING] {rel_path}")
        all_ok = False
        continue

    if rel_path.endswith(".py"):
        with open(full_path, encoding="utf-8") as f:
            source = f.read()
        try:
            ast.parse(source)
            print(f"  [OK]      {rel_path}")
        except SyntaxError as e:
            print(f"  [SYNTAX ERROR] {rel_path}: {e}")
            all_ok = False
    else:
        print(f"  [OK]      {rel_path}  (non-python, exists)")

print()
if all_ok:
    print("All files present and syntax-valid!")
    print()
    print("To run the experiment, first install dependencies:")
    print("  pip install -r requirements.txt")
    print()
    print("Then run:")
    print("  python run_probing.py")
    print("  python run_probing.py --reuse_features   # after first run")
    print("  python run_probing.py --no_cka --no_tsne # fast run")
else:
    print("Some issues found — see above.")
    sys.exit(1)
