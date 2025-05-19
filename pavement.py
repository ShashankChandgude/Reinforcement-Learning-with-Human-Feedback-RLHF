# pavement.py

import os
import shutil
import glob
from paver.easy import task, needs, sh

def rm_rf(pattern):
    """
    Remove files or directories matching the glob pattern.
    Works cross-platform.
    """
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            shutil.rmtree(path)
        elif os.path.isfile(path):
            os.remove(path)

@task
def setup():
    """Install project dependencies."""
    sh("pip install -r requirements.txt")

@task
@needs('setup')
def test():
    """Run the pytest test suite."""
    sh("pytest --maxfail=1 --disable-warnings -q")

@task
def clean():
    """Remove generated artifacts."""
    patterns = [
        "build", "dist", "__pycache__", ".pytest_cache",
        "models", "logs", "evaluation/*.json"
    ]
    for pat in patterns:
        rm_rf(pat)

@task
@needs('test')
def run():
    """Train and evaluate the model end-to-end."""
    sh("./run_all.sh")
