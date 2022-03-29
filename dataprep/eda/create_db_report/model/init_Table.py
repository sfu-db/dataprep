from .init_database import Database
from .init_tablecolumn import TableColumn

class Table:
    def __init__(self, db:Database, schema:str, name:str) -> None:
        self.database = db
        self.schema = schema
        self.name = name
        self.foreign_keys = {}
        self.columns = {}
        self.primary_keys = []
        self.indexes = {}
        self.id = None
        self.checkConstraints = {}
        self.numRows = 0
        self.numCols = 0
        self.comments = None
        self.maxChildren = 0
        self.maxParents = 0
        self.type = None

    def setColumns(self, columns):
        self.columns.update(columns)

    def setIndex(self, name, index):
        self.indexes[name] = index

    def num_row(self, row_size):
        self.numRows = row_size

    def num_columns(self, col_size):
        self.numCols = col_size

    def getIndexes(self):
        return self.indexes.values()

    def getIndex(self, index_name):
        return self.indexes[index_name]

    def getColumns(self):
        return self.columns.values()

    def getPrimaryColumns(self):
        return self.primary_keys

    def setPrimaryColumn(self, primaryKey):
        self.primary_keys.append(primaryKey)

    def addColumn(self, col_name, col):
        self.columns[col_name] = col

    def addedParent(self):
        self.maxParents += 1

    def getMaxParents(self):
        return self.maxParents

    def addedChild(self):
        self.maxChildren += 1

    def getMaxChildren(self):
        return self.maxChildren

    def getViewDefinition(self):
        return None

    def is_view(self):
        return False

    def getType(self):
        if self.is_view() == True:
            return "View"
        return "Table"

    def getColumn(self, colName):
        return self.columns[colName]

    def getName(self):
        return self.name

    def getCheckConstraints(self):
        return self.checkConstraints

    def addCheckConstraint(self, constraintName, text):
        self.checkConstraints[constraintName] = text

    def getForeignKeys(self):
        return self.foreign_keys.values()

    def getColumnsMap(self):
        return self.columns

    def getForeignKeysMap(self):
        return self.foreign_keys

    def addForeignKey(self, foreignkey):
        self.foreign_keys[foreignkey.getName()] = foreignkey
