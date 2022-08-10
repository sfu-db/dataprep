from .table import Table
from .constraint import ForeignKeyConstraint


class TableColumn:
    def __init__(
        self,
        table: Table,
        name: str,
        type_name: str,
        not_null: bool,
        default_value: str,
        is_auto_updated: bool,
        comments: str,
    ):
        self.table = table
        self.name = name.replace("'", "")
        self.type_name = type_name
        self.not_null = not_null
        self.default_value = default_value.replace("'", "") if default_value else ""
        self.comments = comments.replace("'", "") if comments else ""
        self.parents = {}
        self.children = {}
        self.type = 0
        self.decimal_digits = 0
        self.detailed_size = ""
        self.is_auto_updated = is_auto_updated
        self.index = False

    def set_index(self):
        self.index = True

    def is_primary(self):
        if self.table.primary_keys is not None:
            return self in self.table.primary_keys
        return False

    def is_foreign_key(self):
        return len(self.parents) != 0

    def get_default_value(self):
        return self.default_value

    def set_comments(self, comments: str):
        if comments is None or len(comments.strip()) == 0:
            self.comments = None
        else:
            self.comments = comments.strip()

    def add_parent(self, parent, constraint: ForeignKeyConstraint):
        self.parents[parent] = constraint
        self.table.add_max_parents()

    def add_child(self, child, constraint: ForeignKeyConstraint):
        self.children[child] = constraint
        self.table.add_max_children()

    def get_parents(self):
        return list(self.parents.values())

    def get_children(self):
        return list(self.children.values())

    def get_name(self):
        return self.name
