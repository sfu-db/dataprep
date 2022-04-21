from ..db_models.table import Table


class PystacheTable:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.comments = ""
