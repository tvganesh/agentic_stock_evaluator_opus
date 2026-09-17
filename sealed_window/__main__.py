"""``python -m sealed_window`` entry point; delegates to :func:`sealed_window.cli.main`."""

from .cli import main

raise SystemExit(main())
