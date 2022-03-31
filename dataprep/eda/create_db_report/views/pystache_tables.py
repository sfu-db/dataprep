from ..model.init_table import Table


class PSTable:
    def __init__(self, table: Table) -> None:
        self.table = table
        self.comments = ""

    def getTable(self):
        return self.table

    def setTable(self, table: Table):
        self.table = table

    def getComments(self):
        return self.comments

    def setComments(self, comments: str):
        self.comments = comments
