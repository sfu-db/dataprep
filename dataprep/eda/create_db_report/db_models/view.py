from .table import Table
from .database import Database


class View(Table):
    def __init__(self, db: Database, schema: str, name: str, view_def: str) -> None:
        self.viewDefinition = None
        super().__init__(db, schema, name)
        self.set_view_definition(view_def)

    def set_view_definition(self, view_definition: str):
        if view_definition is not None and len(view_definition.strip()) > 0:
            self.viewDefinition = view_definition

    def get_view_definition(self):
        return self.viewDefinition

    def is_view(self):
        return True
