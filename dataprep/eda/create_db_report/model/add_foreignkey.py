from .init_table import Table
from .init_tablecolumn import TableColumn


class ForeignKeyConstraint:

    importedKeyCascade = 0
    importedKeyRestrict = 1
    importedKeySetNull = 2
    importedKeyNoAction = 3

    def __init__(self, child: Table, name: str, deleteRule: str, updateRule: int):
        self.name = name
        self.childTable = child
        self.deleteRule = deleteRule
        self.updateRule = updateRule
        self.parentColumns = []
        self.childColumns = []
        self.parentTable = None

    def addParentColumn(self, column: TableColumn):
        if column != None:
            self.parentColumns.append(column)
            self.parentTable = column.getTable()

    def addChildColumn(self, column: TableColumn):
        if column != None:
            self.childColumns.append(column)

    def getParentColumns(self):
        return self.parentColumns

    def getChildColumns(self):
        return self.childColumns

    def getChildTable(self):
        return self.childTable

    def getParentTable(self):
        return self.parentTable

    def getName(self):
        return self.name

    def getDeleteRule(self):
        return self.deleteRule

    def isCascadeOnDelete(self):
        return self.getDeleteRule() == self.importedKeyCascade

    def isRestrictDelete(self):
        return (
            self.getDeleteRule() == self.importedKeyNoAction
            or self.getDeleteRule() == self.importedKeyRestrict
        )

    def isNullOnDelete(self):
        return self.getDeleteRule() == self.importedKeySetNull

    def getDeleteRuleName(self):
        if self.getDeleteRule() == self.importedKeyCascade:
            return "Cascade on delete"
        elif (
            self.getDeleteRule() == self.importedKeyRestrict
            or self.getDeleteRule() == self.importedKeyNoAction
        ):
            return "Restrict delete"
        elif self.getDeleteRule() == self.importedKeySetNull:
            return "Null on delete"
        else:
            return ""

    def getDeleteRuleDescription(self):
        if self.getDeleteRule() == self.importedKeyCascade:
            return "Cascade on delete:\nDeletion of parent deletes child"
        elif (
            self.getDeleteRule() == self.importedKeyRestrict
            or self.getDeleteRule() == self.importedKeyNoAction
        ):
            return "Restrict delete:\nParent cannot be deleted if children exist"
        elif self.getDeleteRule() == self.importedKeySetNull:
            return "Null on delete:\nForeign key to parent set to NULL when parent deleted"
        else:
            return ""

    def getDeleteRuleAlias(self):
        if self.getDeleteRule() == self.importedKeyCascade:
            return "C"
        elif (
            self.getDeleteRule() == self.importedKeyRestrict
            or self.getDeleteRule() == self.importedKeyNoAction
        ):
            return "R"
        elif self.getDeleteRule() == self.importedKeySetNull:
            return "N"
        else:
            return ""

    def getAllForeignKeyConstraints(tables):
        constraints = []

        for table in tables:
            constraints.extend(table.getForeignKeys())

        return constraints
