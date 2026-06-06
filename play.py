import os
import sys
import runpy

import proj1

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

PLAY_DIR = os.path.join(
    THIS_DIR,
    "scripts",
    "reinforcement_learning",
    "rsl_rl",
)

PLAY_SCRIPT = os.path.join(PLAY_DIR, "play.py")

if PLAY_DIR not in sys.path:
    sys.path.insert(0, PLAY_DIR)

sys.argv = ["play.py"] + sys.argv[1:]

runpy.run_path(PLAY_SCRIPT, run_name="__main__")
