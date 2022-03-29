

class PSTable:
    def __init__(self, table, imageFile) -> None:
        self.table = table
        self.diagramName = imageFile

    def getTable(self):
        return self.table

    def setTable(self, table):
        self.table = table

    def getDiagramName(self):
        return self.diagramName

    def setDiagramName(self, diagramName):
        self.diagramName = diagramName

    def getComments(self):
        return self.comments

    def setComments(self, comments):
        self.comments = comments
