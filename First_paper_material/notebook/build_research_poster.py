"""Build the current poster matching the supplied department template."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).with_name("build_department_poster.py")), run_name="__main__")
