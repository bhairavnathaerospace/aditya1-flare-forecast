"""SoLEXHEL-Net mission console: follow the pipeline and training, replay alerts, start jobs.

    "Training Console.exe" in the project folder (this file, packaged), or
    pythonw dashboard/mission_control.pyw [--tab pipeline|training|watch] [--run outputs/model]

The window lives in dashboard/console/ (see its __init__ for the parts).
"""

import sys
from pathlib import Path

if not getattr(sys, "frozen", False):
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent)]

from console.app import main  # noqa: E402

if __name__ == "__main__":
    main()
