"""
Functions for config downloading and maintaining
"""
from pathlib import Path
from tempfile import gettempdir
from json import dump as jdump

import requests

META_URL = (
    "https://raw.githubusercontent.com/sfu-db/DataConnectorConfigs/master/{}/_meta.json"
)
TABLE_URL = (
    "https://raw.githubusercontent.com/sfu-db/DataConnectorConfigs/master/{}/{}.json"
)


def config_directory() -> Path:
    """
    Returns the config directory path
    """
    tmp = gettempdir()
    return Path(tmp) / "dataprep" / "data_connector"


def ensure_config(impdb: str) -> bool:
    """
    Ensure the config for `impdb` is downloaded
    """
    path = config_directory()
    if (path / impdb).exists():
        return True
    else:
        download_config(impdb)
        return False


def download_config(impdb: str) -> None:
    """
    Download the config from Github into the temp directory.
    """
    url = META_URL.format(impdb)
    meta = requests.get(url).json()
    tables = meta["tables"]

    configs = {"_meta": meta}
    for table in tables:
        url = TABLE_URL.format(impdb, table)
        config = requests.get(url).json()
        configs[table] = config

    path = config_directory()

    (path / impdb).mkdir(parents=True)
    for fname, json in configs.items():
        with (path / impdb / f"{fname}.json").open("w") as f:
            jdump(json, f)
