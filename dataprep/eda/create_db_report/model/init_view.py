from model.init_Table import Table


class view(Table):
    def __init__(self, db, schema, name, view_def) -> None:
        self.viewDefinition = None
        super().__init__(db, schema, name)
        self.setViewDefinition(view_def)

    def setViewDefinition(self, viewDefinition):
        if viewDefinition != None and len(viewDefinition.strip()) > 0:
            self.viewDefinition = viewDefinition

    def getViewDefinition(self):
        return self.viewDefinition

    def is_view(self):
        return True

    def getTable(self):
        return self.Table
