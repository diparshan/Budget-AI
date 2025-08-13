import subprocess
import sys
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def run_script(script_name, *args):
    cmd = [sys.executable, script_name] + list(args)
    print(f"\nRunning: {" ".join(cmd)}")
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    print(process.stdout)
    if process.stderr:
        print(f"Error in {script_name}:\n{process.stderr}")
    process.check_returncode()

if __name__ == "__main__":
    print("Starting model retraining process...")
    try:
        # Retrain Deep Learning Model
        run_script("preprocess_data.py")
        run_script("train_model.py")

        print("All models retrained successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Model retraining failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: Required script not found. Please ensure all scripts are in the same directory: {e}")
        sys.exit(1)


