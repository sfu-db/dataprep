from pickle import NONE

from MySQLdb import STRING
from markupsafe import string


class Schema:

    name = None
    comment = None

    def __init__(self, name: STRING, comment: string) -> None:
        self.name = name
        self.comment = comment
