from matplotlib.pyplot import table


class MustacheTableColumn:
    def __init__(self, tableColumn, indexColumn, rootPath) -> None:
        self.tableColumn = tableColumn
        self.indexColumn = indexColumn
        self.rootPath = rootPath

    def getColumn(self):
        return self.tableColumn

    def getin(self):
        return self.indexColumn

    def getKey(self):
        keyType = ""
        if self.tableColumn.isPrimary():
            keyType = " class='primaryKey' title='Primary Key'"
        elif self.tableColumn.isForeignKey():
            keyType = " class='foreignKey' title='Foreign Key'"
        elif self.getin():
            keyType = " class='" + self.markAsIndexColumn() + "' title='Indexed'"

        return keyType

    def getKeyTitle(self):
        keyTitle = ""
        if self.tableColumn.isPrimary():
            keyTitle = "Primary Key"
        elif self.tableColumn.isForeignKey():
            keyTitle = "Foreign Key"
        elif self.getin():
            keyType = " class='" + self.markAsIndexColumn() + "' title='Indexed'"

        return keyTitle

    def getKeyClass(self):
        keyClass = ""
        if self.tableColumn.isPrimary():
            keyClass = "primaryKey"
        elif self.tableColumn.isForeignKey():
            keyClass = "foreignKey"
        elif self.getin():
            keyType = " class='" + self.markAsIndexColumn() + "' title='Indexed'"

        return keyClass

    def getKeyIcon(self):
        keyIcon = ""
        if self.tableColumn.isPrimary() or self.tableColumn.isForeignKey():
            keyIcon = "<i class='icon ion-key iconkey' style='padding-left: 5px;'></i>"
        elif self.getin():
            keyIcon = "<i class='fa fa-sitemap fa-rotate-120' style='padding-right: 5px;'></i>"

        return keyIcon

    def getNullable(self):
        if not self.tableColumn.notnull():
            return "√"
        else:
            return ""

    def getTitleNullable(self):
        if self.tableColumn.notnull():
            return "nullable"
        else:
            return ""

    def getAutoUpdated(self):
        if self.tableColumn.isautoupdated():
            return "√"
        else:
            return ""

    def getTitleAutoUpdated(self):
        if self.tableColumn.isautoupdated():
            return "Automatically updated by the database"
        else:
            return ""

    def markAsIndexColumn(self):
        if self.indexColumn:
            print("yeas")
            return "indexedColumn"
        else:
            return ""

    def getDefaultValue(self):
        return str(self.tableColumn.getDefaultValue())

    def markAsIndexColumn(self):
        if self.indexColumn:
            return "indexedColumn"
        return ""
