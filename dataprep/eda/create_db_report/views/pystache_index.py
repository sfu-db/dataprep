class PSIndex:
    def __init__(self, index) -> None:
        self.index = index

    def getIndex(self):
        return self.index

    def getKey(self):
        if self.index.isPrimaryKey():
            keyType = " class='primaryKey' title='Primary Key'"
        elif self.index.isUniqueKey():
            keyType = " class='uniqueKey' title='Unique Key'"
        else:
            keyType = " title='Indexed'"
        return keyType

    def getKeyIcon(self):
        keyIcon = ""
        if self.index.isPrimaryKey() or self.index.isUniqueKey():
            keyIcon = "<i class='icon ion-key iconkey'></i> "

        return keyIcon
