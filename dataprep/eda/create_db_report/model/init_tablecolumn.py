from .init_Table import Table

class TableColumn:
    def __init__(self, table:Table, name:str, typeName:str, notNull:bool, defaultValue:str, isAutoUpdated:bool, comments:str):
        self.table = table
        self.name = name
        self.typeName = typeName
        self.notNull = notNull
        self.defaultValue = defaultValue
        self.comments = comments
        self.parents = {}
        self.children = {}
        self.type = 0
        self.decimalDigits = 0
        self.detailedSize = None
        self.isAutoUpdated = isAutoUpdated
        self.index = False

    def setIndex(self):
        self.index = True

    def isIndexCol(self):
        return self.index

    def getTable(self):
        return self.table

    def getName(self):
        return self.name

    def isautoupdated(self):
        return self.isAutoUpdated

    def notnull(self):
        return self.notNull

    def getTypeName(self):
        return self.typeName

    def isPrimary(self):
        if self.table.getPrimaryColumns() != None:
            return self in self.table.getPrimaryColumns()
        return False

    def isForeignKey(self):
        return len(self.parents) != 0

    def getDefaultValue(self):
        return self.defaultValue

    def setDefaultValue(self, defaultValue):
        self.defaultValue = defaultValue

    def getComments(self):
        return self.comments

    def setComments(self, comments):
        if comments == None or len(comments.strip()) == 0:
            self.comments = None
        else:
            self.comments = comments.strip()

    def addParent(self, parent, constraint):
        self.parents[parent] = constraint
        self.table.addedParent()

    def addChild(self, child, constraint):
        self.children[child] = constraint
        self.table.addedChild()

    def getParents(self):
        return list(self.parents)

    def getChildren(self):
        return list(self.children)
