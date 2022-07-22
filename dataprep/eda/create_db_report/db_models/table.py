from .database import Database
from .table_index import TableIndex


class Table:
    def __init__(self, database: Database, schema: str, name: str) -> None:
        self.database = database
        self.schema = schema.replace("'", "")
        self.name = name.replace("'", "")
        self.foreign_keys = {}
        self.columns = {}
        self.primary_keys = []
        self.indexes = {}
        self.referenced_by = {}
        self.id = None
        self.check_constraints = {}
        self.num_of_rows = 0
        self.num_of_cols = 0
        self.comments = None
        self.max_children = 0
        self.max_parents = 0
        self.type = None

    def set_columns(self, columns):
        self.columns.update(columns)

    def set_index(self, name: str, index: TableIndex):
        self.indexes[name] = index

    def num_columns(self, col_size: int):
        self.num_of_cols = col_size

    def get_indexes(self):
        return self.indexes.values()

    def get_index(self, index_name: str):
        return self.indexes[index_name]

    def get_columns(self):
        return self.columns.values()

    def add_primary_key(self, primary_key):
        self.primary_keys.append(primary_key)

    def add_column(self, col_name: str, col):
        self.columns[col_name] = col

    def add_max_parents(self):
        self.max_parents += 1

    def add_max_children(self):
        self.max_children += 1

    def add_referenced_by_table(self, table):
        self.referenced_by[table.get_name()] = table

    def get_referenced_by_tables(self):
        return self.referenced_by

    def get_view_definition(self):
        return None

    def is_view(self):
        return False

    def get_type(self):
        if self.is_view():
            return "View"
        return "Table"

    def get_column(self, col_name: str):
        return self.columns[col_name]

    def add_check_constraint(self, constraint_name: str, text: str):
        self.check_constraints[constraint_name] = text

    def get_foreign_keys(self):
        return self.foreign_keys.values()

    def get_foreign_keys_dict(self):
        return self.foreign_keys

    def add_foreign_key(self, foreign_key):
        self.foreign_keys[foreign_key.name] = foreign_key

    def get_name(self):
        return self.name
