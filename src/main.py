# FileName: main.py
# Description: This script automates the project setup and execution. It creates a
#              virtual environment, installs dependencies from requirements.txt, and
#              runs the data processing, feature extraction, and model training
#              scripts in the correct sequence.

import os
import sys
import subprocess
import venv

# --- Project Configuration ---
VENV_DIR = "venv"
REQUIREMENTS_FILE = "requirements.txt"
# Defines the sequence of scripts to execute for the main pipeline.
PYTHON_SCRIPTS = [
    "src/data_preparation.py",
    "src/feature_extraction.py",
    "src/train_classifier.py"
]
APP_SCRIPT = "src/app.py"


def run_command(command, venv_python_path=None):
    """Runs a shell command and prints its output in real-time."""
    print(f"\n--- Running command: {' '.join(command)} ---")
    try:
        # If a python executable path is provided, use it to run python commands.
        # This ensures commands are executed within the project's virtual environment.
        if venv_python_path and command[0] == "python":
            command[0] = venv_python_path
            
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        for line in process.stdout:
            print(line, end="")
        process.wait()
        if process.returncode != 0:
            print(f"--- !!! Command failed with error code: {process.returncode} !!! ---")
            return False
    except Exception as e:
        print(f"--- !!! An exception occurred: {e} !!! ---")
        return False
    print(f"--- Command finished successfully ---")
    return True


def get_venv_paths():
    """Returns the platform-specific paths for the virtual environment's executables."""
    # Handles differences in directory structure between Windows and Unix-like systems.
    if sys.platform == "win32":
        venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
        venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    else:
        venv_python = os.path.join(VENV_DIR, "bin", "python")
        venv_pip = os.path.join(VENV_DIR, "bin", "pip")
    return venv_python, venv_pip


def create_virtual_env():
    """Creates a virtual environment in the specified directory if it doesn't exist."""
    if not os.path.exists(VENV_DIR):
        print(f"Virtual environment '{VENV_DIR}' not found. Creating...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            print("Virtual environment created successfully.")
        except Exception as e:
            print(f"Could not create virtual environment: {e}")
            return False
    else:
        print(f"Virtual environment '{VENV_DIR}' already exists.")
    return True


def install_requirements(venv_pip_path):
    """Installs Python packages from the requirements.txt file."""
    if not os.path.exists(REQUIREMENTS_FILE):
        print(f"ERROR: '{REQUIREMENTS_FILE}' not found. Cannot install dependencies.")
        return False
        
    print("Installing dependencies from requirements.txt...")
    return run_command([venv_pip_path, "install", "-r", REQUIREMENTS_FILE])


def run_pipeline():
    """The main function to orchestrate the entire project setup and execution."""
    print(">>> STARTING PROJECT AUTOMATION SCRIPT <<<")

    # Create the virtual environment.
    if not create_virtual_env():
        sys.exit(1)

    venv_python_path, venv_pip_path = get_venv_paths()

    # Install all required dependencies into the virtual environment.
    if not install_requirements(venv_pip_path):
        print("Halting due to failed dependency installation.")
        sys.exit(1)

    # Execute the core project scripts in their designated order.
    print("\n>>> EXECUTING PROJECT PHASES <<<")
    for script_path in PYTHON_SCRIPTS:
        if not run_command(["python", script_path], venv_python_path=venv_python_path):
            print(f"Execution failed at script: {script_path}. Halting.")
            sys.exit(1)
            
    # Provide final instructions to the user for launching the web application.
    print("\n>>> AUTOMATION COMPLETE <<<")
    print("All preliminary phases have been executed successfully.")
    print("To start the interactive web dashboard, please run the following command in your terminal:")
    print(f"\n  {venv_python_path} {APP_SCRIPT}\n")


if __name__ == "__main__":
    run_pipeline()