"""Hugging Face Spaces entry point.

Spaces runs `app.py` at the repository root, so this is a two-line shim over
the real UI in `vichara.ui.app` rather than a second implementation. One code
path for the Space, the CLI and the tests.
"""

import os

# Belt and braces with analytics_enabled=False in the Blocks constructor:
# Gradio checks this before the app object exists, so setting it here covers
# the import-time probe as well as the launch-time one.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

from vichara.ui.app import main

if __name__ == "__main__":
    main()
