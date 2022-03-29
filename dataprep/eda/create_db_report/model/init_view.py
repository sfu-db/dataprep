from .init_table import Table
from .init_database import Database


class View(Table):
    def __init__(self, db: Database, schema: str, name: str, view_def: str) -> None:
        self.viewDefinition = None
        super().__init__(db, schema, name)
        self.setViewDefinition(view_def)

    def setViewDefinition(self, viewDefinition):
        if viewDefinition != None and len(viewDefinition.strip()) > 0:
            self.viewDefinition = viewDefinition

    def getViewDefinition(self):
        return self.viewDefinition

    def is_view(self):
        return True
