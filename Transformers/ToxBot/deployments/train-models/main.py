import subprocess

scripts = ["bin/train_generator.py", "bin/train_classifier.py"]

for s in scripts:
    subprocess.call(["python", s])