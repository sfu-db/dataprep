class PSConstraints:
    tableName = ""
    name = ""
    definition = ""

    def __init__(self, tableName, name, definition) -> None:
        self.tableName = tableName
        self.name = name
        self.definition = definition

    def getTableName(self):
        return self.tableName

    def getName(self):
        return self.name

    def getDefinition(self):
        return self.definition
