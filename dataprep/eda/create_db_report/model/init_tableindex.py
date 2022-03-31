class TableIndex:
    def __init__(self, name: str, indexType: str) -> None:
        self.name = name
        self.isUnique = False
        self.isPrimary = False
        self.columns = {}
        self.indexType = indexType

    def getName(self):
        return self.name

    def addColumn(self, col_string: str, column):
        if column != None:
            self.columns[col_string] = column

    def setPrimary(self):
        self.isPrimary = True

    def setUnique(self):
        self.isUnique = True

    def isPrimaryKey(self):
        return self.isPrimary

    def isUniqueKey(self):
        return self.isUnique

    def getType(self):
        if self.isPrimaryKey():
            return "Primary key"
        if self.isUniqueKey():
            return "Must be unique"
        return "Performance"

    def getIndexType(self):
        return self.indexType

    def columnsAsString(self):
        return "".join(self.columns.keys())
