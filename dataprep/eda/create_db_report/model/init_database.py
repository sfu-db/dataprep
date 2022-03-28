from tokenize import String
from .init_Table import Table


class Database:
    def __init__(self, databaseName, schema, database_stats) -> None:
        self.databaseName = databaseName
        self.schema = schema
        self.tables = {}
        self.views = {}
        for key, value in database_stats.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def addTable(self, table_name: String, table_object: Table):
        self.tables[table_name] = table_object

    def addView(self, view_name, view_object):
        self.views[view_name] = view_object

    def getTablesMap(self):
        return self.tables

    def getViewsMap(self):
        return self.views

    def getTableNames(self):
        return self.tables.keys()

    def getName(self):
        return self.databaseName

    def getTables(self):
        return self.tables.values()

    def getSchema(self):
        return self.schema

    def getViews(self):
        return self.views.values()

    def getViewsMap(self):
        return self.views
