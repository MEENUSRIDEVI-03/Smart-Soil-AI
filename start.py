import os, subprocess, sys
folder = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder)
subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"], check=True)
