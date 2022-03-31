class PSConstraints:
    def __init__(self, tableName: str, name: str, definition: str) -> None:
        self.tableName = tableName
        self.name = name
        self.definition = definition

    def getTableName(self):
        return self.tableName

    def getName(self):
        return self.name

    def getDefinition(self):
        return self.definition
