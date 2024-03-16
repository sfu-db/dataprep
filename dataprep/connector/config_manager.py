"""
Functions for config downloading and maintaining
"""

import json
from json import dump as jdump
from pathlib import Path
from shutil import rmtree
from tempfile import gettempdir
from typing import cast, Tuple

from .utils import Request

# note: apply change after rename the config repo
META_URL = "https://raw.githubusercontent.com/sfu-db/APIConnectors/{}/api-connectors/{}/_meta.json"
TABLE_URL = "https://raw.githubusercontent.com/sfu-db/APIConnectors/{}/api-connectors/{}/{}.json"
GIT_REF_URL = "https://api.github.com/repos/sfu-db/APIConnectors/git/refs/heads"


def separate_branch(config_path: str) -> Tuple[str, str]:
    """Separate the config path into db name and branch"""
    segments = config_path.split("@")
    if len(segments) == 1:
        return segments[0], "master"
    elif len(segments) == 2:
        return segments[0], segments[1]
    else:
        raise ValueError(f"Multiple branches in the config path {config_path}")


def initialize_path(config_path: str, update: bool) -> Path:
    """Determines if the given config_path is local or in GitHub.
    Fetches the full path."""
    if config_path.startswith(".") or config_path.startswith("/") or config_path.startswith("~"):
        path = Path(config_path).resolve()
    else:
        # From GitHub!
        impdb, branch = separate_branch(config_path)
        ensure_config(impdb, branch, update)
        path = config_directory() / branch / impdb
    return path


def config_directory() -> Path:
    """
    Returns the config directory path
    """
    tmp = gettempdir()
    return Path(tmp) / "dataprep" / "connector"


def ensure_config(impdb: str, branch: str, update: bool) -> bool:
    """Ensure the config for `impdb` is downloaded"""
    path = config_directory()

    if (path / branch / impdb / "_meta.json").exists() and not update:
        return True

    obsolete = is_obsolete(impdb, branch)

    if (path / branch / impdb / "_meta.json").exists() and not obsolete:
        return True
    else:
        download_config(impdb, branch)
        return False


def is_obsolete(impdb: str, branch: str) -> bool:
    """Test if the implicit db config files are obsolete and need to be re-downloaded."""

    path = config_directory()
    if not (path / branch / impdb / "_meta.json").exists():
        return True
    elif not (path / branch / impdb / "_hash").exists():
        return True
    else:
        with open(path / branch / impdb / "_hash", "r") as f:
            githash = f.read()

        sha = get_git_branch_hash(branch)

        return githash != sha


def get_git_branch_hash(branch: str) -> str:
    """Get current config files repo's hash"""
    requests = Request(GIT_REF_URL)
    response = requests.get()
    refs = json.loads(response.read())

    (sha,) = [ref["object"]["sha"] for ref in refs if ref["ref"] == f"refs/heads/{branch}"]
    return cast(str, sha)


def download_config(impdb: str, branch: str) -> None:
    """Download the config from Github into the temp directory."""
    requests = Request(META_URL.format(branch, impdb))
    response = requests.get()
    meta = json.loads(response.read())
    tables = meta["tables"]

    sha = get_git_branch_hash(branch)
    # In case we push a new config version to github when the user is downloading
    while True:
        configs = {"_meta": meta}
        for table in tables:
            requests = Request(TABLE_URL.format(branch, impdb, table))
            response = requests.get()
            config = json.loads(response.read())
            configs[table] = config
        sha_check = get_git_branch_hash(branch)

        if sha_check == sha:
            break

        sha = sha_check

    path = config_directory()

    if (path / branch / impdb).exists():
        rmtree(path / branch / impdb)

    (path / branch / impdb).mkdir(parents=True)
    for fname, val in configs.items():
        with (path / branch / impdb / f"{fname}.json").open("w") as f:
            jdump(val, f)

    with (path / branch / impdb / "_hash").open("w") as f:
        f.write(sha)
