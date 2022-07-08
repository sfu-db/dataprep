class TemplateConstraint:
    def __init__(self, table_name: str, name: str, definition: str) -> None:
        self.table_name = table_name
        self.name = name
        self.definition = definition
