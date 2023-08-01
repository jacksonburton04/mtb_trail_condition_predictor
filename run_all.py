import subprocess
import sys

scripts = ["01_mtb_prep_data.py", "03_make_preds.py", "04_visualize.py"]

if "run_model" in sys.argv:
    scripts.insert(1, "02_mtb_build_model.py")

for script in scripts:
    subprocess.run(["python3", script])