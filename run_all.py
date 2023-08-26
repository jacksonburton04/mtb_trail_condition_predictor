import logging
import subprocess
import sys

# Initialize logging
logging.basicConfig(filename='run_all.log', level=logging.INFO)

# Log the start time
logging.info('Script started')

# Your existing code
scripts = ["01_mtb_prep_data.py", "03_make_preds.py", "04_visualize.py"]

if "run_model" in sys.argv:
    scripts.insert(1, "02_mtb_build_model.py")

for script in scripts:
    try:
        subprocess.run(["python3", script])
        logging.info(f'Successfully ran {script}')
    except Exception as e:
        logging.error(f'Error running {script}: {e}')

# Log the end time
logging.info('Script ended')
