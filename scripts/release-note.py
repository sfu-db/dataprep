"""ReleaseNote
Usage:
    release-note <hash>
"""

import re
from binascii import hexlify
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from docopt import docopt
from git import Commit
from git.repo import Repo
from jinja2 import Template

VERSION_REG = re.compile(r"^v(\d+)\.(\d+)\.(\d+)")
MESSAGE_REG = re.compile(
    r"^(?P<type>feat|perf|docs|fix|test|chore)\((?P<scope>.+?)\)?: ?(?P<message>.+)"
)

RELEASE_NOTE_TEMPLATE = Template(
    """
{% if fixes %}
## Bugfixes 🐛
{% for fix in fixes -%}
* {{fix.scope}}: {{fix.message}} ({{fix.hash}})
{% endfor %}{% endif %}

{% if feats %}
## Features ✨
{% for feat in feats -%}
* {{feat.scope}}: {{feat.message}} ({{feat.hash}})
{% endfor %}
{% endif %}

{% if tests %}
## Code Quality + Testing 💯
{% for test in tests -%}
* {{test.scope}}: {{test.message}} ({{test.hash}})
{% endfor %}
{% endif %}

{% if perfs %}
## Performance 🚀
{% for perf in perfs -%}
* {{perf.scope}}: {{perf.message}} ({{perf.hash}})
{% endfor %}
{% endif %}

{% if docs %}
## Documentation 📃
{% for doc in docs -%}
* {{doc.scope}}: {{doc.message}} ({{doc.hash}})
{% endfor %}
{% endif %}

{% if chores %}
## Others 🛠
{% for chore in chores -%}
* {{chore.scope}}: {{chore.message}} ({{chore.hash}})
{% endfor %}
{% endif %}


## Contributors this release 🏆

The following users contributed code to DataPrep since the last release.

{% for author in authors -%}
* {{author.name}} \<{{author.email}}\> {% if author.first %}(First time contributor) ⭐️{% endif %}
{% endfor %}

🎉🎉 Thank you! 🎉🎉
"""
)


def main() -> None:
    args = docopt(__doc__)

    repo = Repo(".")
    assert repo.bare == False
    hash = args["<hash>"]
    this_commits, handle = commits_since_previous(repo.commit(hash))

    version = VERSION_REG.match(handle.message).group()

    previous_commits: List[Commit] = []
    while handle is not None:
        commits, handle = commits_since_previous(*handle.parents)
        previous_commits.extend(commits.values())

    this_authors = {commit.author for commit in this_commits.values()}

    first_time_authors = this_authors - {commit.author for commit in previous_commits}

    authors = [
        {
            "name": author.name,
            "email": author.email,
            "first": author in first_time_authors,
        }
        for author in this_authors
    ]

    authors = sorted(authors, key=lambda a: a["name"].upper())

    notes = defaultdict(list)
    for commit in this_commits.values():
        match = MESSAGE_REG.match(commit.message)
        if match is not None:
            notes[match.group("type")].append(
                {
                    "scope": match.group("scope"),
                    "message": match.group("message"),
                    "hash": hexlify(commit.binsha).decode()[:8],
                }
            )

    note = RELEASE_NOTE_TEMPLATE.render(
        feats=notes["feat"],
        perfs=notes["perf"],
        docs=notes["docs"],
        chore=notes["chore"],
        tests=notes["test"],
        fixes=notes["fix"],
        authors=authors,
    )

    new_note = note.replace("\n\n\n", "\n\n")
    while new_note != note:
        note = new_note
        new_note = note.replace("\n\n\n", "\n\n")
    note = new_note

    note = note.strip()

    print(note)


def commits_since_previous(
    *seed_commits: Commit,
) -> Tuple[Dict[str, Commit], Optional[Commit]]:
    stack = list(seed_commits)
    commits = {}

    previous = None
    while stack:
        commit = stack.pop()

        if commit.binsha in commits:
            continue

        matches = VERSION_REG.findall(commit.message)

        if matches:
            previous = commit
            continue

        commits[commit.binsha] = commit
        stack.extend(commit.parents)

    return commits, previous


def find_commit_by_hash(seed: Commit, hash: str) -> Optional[Commit]:
    stack = [seed]
    commits = {}

    while stack:
        commit = stack.pop()

        if commit.binsha in commits:
            continue
        hex = hexlify(commit.binsha)
        if hex.startswith(hash.encode()):
            return commit

        commits[commit.binsha] = commit
        stack.extend(commit.parents)

    return None


if __name__ == "__main__":
    main()
