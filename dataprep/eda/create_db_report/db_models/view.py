from .table import Table
from .database import Database


class View(Table):
    def __init__(self, db: Database, schema: str, name: str, view_def: str) -> None:
        self.view_definition = None
        super().__init__(db, schema, name)
        self.set_view_definition(view_def)

    def set_view_definition(self, view_definition: str):
        if view_definition is not None and len(view_definition.strip()) > 0:
            self.view_definition = view_definition

    def get_view_definition(self):
        return self.view_definition

    def is_view(self):
        return True
