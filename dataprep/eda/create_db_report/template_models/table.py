from ..db_models.table import Table


class TemplateTable:
    def __init__(self, table: Table) -> None:
        for attr in dir(table):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(table, attr))
        self.table = table
        self.comments = ""
