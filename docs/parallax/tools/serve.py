#!/usr/bin/env python3
"""Preview the docs site, or run its strict build with --build. Needs only Python: uses uvx
when present, otherwise a private venv in docs/parallax/.venv."""
import os
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

PARALLAX = Path(__file__).resolve().parents[1]
CONFIG = PARALLAX / "mkdocs.yml"
PINS = PARALLAX / "requirements.txt"
VENV = PARALLAX / ".venv"
BIN = VENV / ("Scripts" if os.name == "nt" else "bin")


def mkdocs_command():
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--with-requirements", str(PINS), "mkdocs"]
    stamp = VENV / "requirements.txt"
    if not stamp.is_file() or stamp.read_text() != PINS.read_text():
        print("First run: installing the docs tools into docs/parallax/.venv", file=sys.stderr)
        venv.create(VENV, with_pip=True, clear=True)
        subprocess.run([str(BIN / "python"), "-m", "pip", "install", "-q", "-r", str(PINS)], check=True)
        stamp.write_text(PINS.read_text())
    return [str(BIN / "mkdocs")]


def main():
    args = sys.argv[1:]
    build = "--build" in args
    extra = [a for a in args if a != "--build"]
    mode = ["build", "--strict"] if build else ["serve"]
    if build and "-d" not in extra:
        extra += ["-d", tempfile.mkdtemp(prefix="parallax-site-")]
    command = mkdocs_command() + mode + ["-f", str(CONFIG)] + extra
    try:
        return subprocess.run(command, env={**os.environ, "NO_MKDOCS_2_WARNING": "1"}).returncode
    except KeyboardInterrupt:
        return 0
    except FileNotFoundError:
        print(f"Could not run {command[0]}: delete docs/parallax/.venv and try again", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
