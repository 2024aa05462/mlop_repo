#!/usr/bin/env python3
"""
Execute Jupyter notebooks and save with outputs.
"""
import os
import sys
import nbformat
from nbclient import NotebookClient

# Change to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

notebooks = [
    'notebooks/01_eda.ipynb',
    'notebooks/02_feature_engineering_modeling.ipynb',
    'notebooks/03_mlflow_experiments.ipynb'
]

print("=" * 70)
print("          EXECUTING NOTEBOOKS WITH OUTPUT CAPTURE")
print("=" * 70)

for nb_path in notebooks:
    full_path = os.path.join(PROJECT_ROOT, nb_path)
    if not os.path.exists(full_path):
        print(f"❌ Not found: {nb_path}")
        continue
    
    print(f"\n📓 Executing: {nb_path}")
    
    try:
        # Read notebook
        with open(full_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Create client and execute
        client = NotebookClient(
            nb,
            timeout=600,
            kernel_name='python3',
            resources={'metadata': {'path': os.path.dirname(full_path)}}
        )
        
        # Execute the notebook
        client.execute()
        
        # Save with outputs
        with open(full_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"   ✅ Executed and saved: {nb_path}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 70)
print("                    EXECUTION COMPLETE!")
print("=" * 70)

