"""
Setup script for MLLM Security Project
Automates environment creation and dependency installation
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print('='*70)
    print(f"Running: {cmd}")
    print()
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode == 0:
        print(f"\n✅ {description} - SUCCESS")
        return True
    else:
        print(f"\n❌ {description} - FAILED")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          MLLM SECURITY PROJECT - SETUP SCRIPT                    ║
╚══════════════════════════════════════════════════════════════════╝

This script will:
  1. Create a Python virtual environment
  2. Install all required dependencies
  3. Verify the installation
  4. Guide you through the next steps
    """)
    
    input("Press Enter to continue...")
    
    # Check Python version
    print(f"\n📌 Python version: {sys.version}")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        return
    
    # Step 1: Create virtual environment
    venv_name = "mllm_env"
    
    if os.path.exists(venv_name):
        print(f"\n⚠️  Virtual environment '{venv_name}' already exists.")
        response = input("Delete and recreate? (yes/no): ").lower()
        if response == 'yes':
            import shutil
            shutil.rmtree(venv_name)
            print(f"✅ Removed existing '{venv_name}'")
        else:
            print("Skipping virtual environment creation...")
    
    if not os.path.exists(venv_name):
        if not run_command(
            f"python -m venv {venv_name}",
            "Creating virtual environment"
        ):
            return
    
    # Determine activation command
    if sys.platform == "win32":
        python_exe = f"{venv_name}\\Scripts\\python.exe"
        pip_exe = f"{venv_name}\\Scripts\\pip.exe"
        activate_cmd = f"{venv_name}\\Scripts\\Activate.ps1"
    else:
        python_exe = f"{venv_name}/bin/python"
        pip_exe = f"{venv_name}/bin/pip"
        activate_cmd = f"source {venv_name}/bin/activate"
    
    # Step 2: Upgrade pip
    run_command(
        f"{python_exe} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Step 3: Install requirements
    if not run_command(
        f"{pip_exe} install -r requirements.txt",
        "Installing dependencies (this may take several minutes)"
    ):
        print("\n⚠️  Some packages failed to install.")
        print("You may need to install them manually.")
    
    # Step 4: Verify installation
    print("\n" + "="*70)
    print("  VERIFYING INSTALLATION")
    print("="*70)
    
    verification_code = """
import sys
packages = ['torch', 'transformers', 'datasets', 'sklearn', 'pandas', 'matplotlib']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg:20s} - installed')
    except ImportError:
        print(f'❌ {pkg:20s} - MISSING')
        missing.append(pkg)

if not missing:
    print('\\n✅ All packages installed successfully!')
    sys.exit(0)
else:
    print(f'\\n❌ Missing packages: {", ".join(missing)}')
    sys.exit(1)
"""
    
    result = subprocess.run(
        [python_exe, "-c", verification_code],
        capture_output=False,
        text=True
    )
    
    # Final instructions
    print("\n" + "="*70)
    print("  SETUP COMPLETE!")
    print("="*70)
    
    print(f"""
✅ Virtual environment created: {venv_name}/

🚀 NEXT STEPS:

1. Activate the environment:
   Windows PowerShell:
     {activate_cmd}
   
   Windows CMD:
     {venv_name}\\Scripts\\activate.bat
   
   Linux/Mac:
     source {venv_name}/bin/activate

2. Login to Hugging Face (required for dataset):
   huggingface-cli login
   
   Get your token from: https://huggingface.co/settings/tokens

3. Run the quick start menu:
   python src\\quickstart.py

4. Or train directly:
   python src\\train_toxicity.py

📚 Documentation:
   - QUICKSTART.md - Quick reference guide
   - README.md - Full documentation

⚡ Tips:
   - Training takes 10-30 minutes on CPU, 5-10 minutes on GPU
   - Reduce batch_size if you run out of memory
   - Model will be saved to models/toxicity_classifier/
""")
    
    print("="*70)
    print("Happy coding! 🎉")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        import traceback
        traceback.print_exc()
