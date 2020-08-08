"""
Module contains the loaded config schema.
"""
from json import load as jload
from pathlib import Path

with open(f"{Path(__file__).parent}/schema.json", "r") as f:
    CONFIG_SCHEMA = jload(f)
