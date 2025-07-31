# main.py
# This script is a simple way to run the whole project. It sets up the
# virtual environment, installs the dependencies, and then runs all the
# data processing and model training scripts.

import os
import sys
import subprocess
import venv

# --- Config ---
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"
PIPELINE_SCRIPTS = [
    "src/data_preparation.py",
    "src/feature_extraction.py",
    "src/train_classifier.py"
]
APP_SCRIPT = "src/app.py"


def run_command(command, venv_python=None):
    """Runs a command and prints output in real-time."""
    print(f"\n--- Running: {' '.join(command)} ---")
    try:
        # If we have a venv, make sure we use it
        if venv_python and command[0] == "python":
            command[0] = venv_python
            
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"--- Command failed! ---")
            return False
    except Exception as e:
        print(f"--- An exception occurred: {e} ---")
        return False
    print(f"--- Done ---")
    return True


def get_venv_paths():
    """Gets the venv paths for the current OS."""
    if sys.platform == "win32":
        # Windows paths
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        # Unix-like paths
        venv_python = os.path.join(VENV_DIR, "bin", "python")
        venv_pip = os.path.join(VENV_DIR, "bin", "pip")
    return venv_python, venv_pip


def setup_venv():
    """Creates a venv if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment at '{VENV_DIR}'...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print("Done.")
        except Exception as e:
            print(f"Couldn't create venv: {e}")
            return False
    else:
        print(f"Virtual environment already exists.")
    return True


def install_requirements(venv_pip):
    """Installs packages from requirements.txt."""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"Error: '{REQUIREMENTS_FILE}' not found.")
        return False
        
    print("Installing dependencies...")
    return run_command([venv_pip, "install", "-r", REQUIREMENTS_FILE])


def main():
    """Runs the whole show."""
    print(">>> Kicking off the project setup <<<")

    # Create venv
    if not setup_venv():
        sys.exit(1)

    venv_python, venv_pip = get_venv_paths()

    # Install dependencies
    if not install_requirements(venv_pip):
        print("Failed to install dependencies. Stopping.")
        sys.exit(1)

    # Run the pipeline
    print("\n>>> Running the pipeline scripts <<<")
    for script in PIPELINE_SCRIPTS:
        if not run_command(["python", script], venv_python=venv_python):
            print(f"Script failed: {script}. Stopping.")
            sys.exit(1)
            
    # All done!
    print("\n>>> All done! <<<")
    print("To start the web app, run this command:")
    print(f"\n  {venv_python} {APP_SCRIPT}\n")


if __name__ == "__main__":
    main()
