#!/usr/bin/env python3
"""
Main entry point for the TIDE 2.0 Unified Interface.

This Streamlit app provides an interactive interface for visualizing,
comparing, and editing detected entities in text data.
"""

import subprocess  # nosec B404
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Main entry point for the CLI.

    Args:
        argv: Command-line arguments. Defaults to ``sys.argv[1:]``.
    """
    args = sys.argv[1:] if argv is None else argv

    # Get the path to the unified_interface.py file
    current_dir = Path(__file__).parent
    streamlit_app = current_dir / "unified_interface.py"

    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)

    cmd = ["streamlit", "run", str(streamlit_app)]
    if args:
        cmd.extend(["--", *args])

    # Launch streamlit with the app file
    try:
        subprocess.run(cmd, check=True)  # noqa: S603 # nosec B603
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install streamlit: uv pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
