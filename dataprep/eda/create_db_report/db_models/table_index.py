class TableIndex:
    def __init__(self, name: str, index_type: str) -> None:
        self.name = name.replace("'", "")
        self.is_unique = False
        self.is_primary = False
        self.columns = {}
        self.index_type = index_type

    def add_column(self, col_string: str, column):
        if column is not None:
            self.columns[col_string] = column

    def set_primary(self):
        self.is_primary = True

    def set_unique(self):
        self.is_unique = True

    def get_type(self):
        if self.is_primary:
            return "Primary key"
        if self.is_unique:
            return "Must be unique"
        return "Performance"

    def get_index_type(self):
        return self.index_type

    def columns_as_string(self):
        return "".join(self.columns.keys())
