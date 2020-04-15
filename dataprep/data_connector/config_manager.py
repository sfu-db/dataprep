"""
Functions for config downloading and maintaining
"""
from json import dump as jdump
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from typing import cast

import requests

META_URL = (
    "https://raw.githubusercontent.com/sfu-db/DataConnectorConfigs/master/{}/_meta.json"
)
TABLE_URL = (
    "https://raw.githubusercontent.com/sfu-db/DataConnectorConfigs/master/{}/{}.json"
)
GIT_REF_URL = "https://api.github.com/repos/sfu-db/DataConnectorConfigs/git/refs/heads"


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
    obsolete = is_obsolete(impdb)

    if (path / impdb).exists() and not obsolete:
        return True
    else:
        download_config(impdb)
        return False


def is_obsolete(impdb: str) -> bool:
    """
    Test if the implicit db config files are obsolete
    and need to be re-downloaded.
    """
    path = config_directory()
    if not (path / impdb).exists():
        return True
    elif not (path / impdb / "_hash").exists():
        return True
    else:
        with open(path / impdb / "_hash", "r") as f:
            githash = f.read()

        sha = get_git_master_hash()

        return githash != sha


def get_git_master_hash() -> str:
    """
    Get current config files repo's hash
    """
    refs = requests.get(GIT_REF_URL).json()
    (sha,) = [ref["object"]["sha"] for ref in refs if ref["ref"] == "refs/heads/master"]
    return cast(str, sha)


def download_config(impdb: str) -> None:
    """
    Download the config from Github into the temp directory.
    """
    url = META_URL.format(impdb)
    meta = requests.get(url).json()
    tables = meta["tables"]

    sha = get_git_master_hash()
    # In case we push a new config version to github when the user is downloading
    while True:
        configs = {"_meta": meta}
        for table in tables:
            url = TABLE_URL.format(impdb, table)
            config = requests.get(url).json()
            configs[table] = config
        sha_check = get_git_master_hash()

        if sha_check == sha:
            break

        sha = sha_check

    path = config_directory()

    if (path / impdb).exists():
        rmtree(path / impdb)

    (path / impdb).mkdir(parents=True)
    for fname, json in configs.items():
        with (path / impdb / f"{fname}.json").open("w") as f:
            jdump(json, f)

    with (path / impdb / "_hash").open("w") as f:
        f.write(sha)
