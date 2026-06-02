#!/usr/bin/env python3
"""
Main entry point for the TIDE 2.0 Unified Interface.

This Streamlit app provides an interactive interface for visualizing,
comparing, and editing detected entities in text data.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Main entry point for the CLI."""
    # Get the path to the unified_interface.py file
    current_dir = Path(__file__).parent
    streamlit_app = current_dir / "unified_interface.py"

    if not streamlit_app.exists():
        print(f"Error: Streamlit app not found at {streamlit_app}")
        sys.exit(1)

    # Launch streamlit with the app file
    try:
        subprocess.run(["streamlit", "run", str(streamlit_app)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install streamlit: pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
