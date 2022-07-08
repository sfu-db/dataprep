from ..db_models.table_column import TableColumn


class TemplateTableColumn:
    def __init__(self, table_column: TableColumn, index_column: bool, root_path: str) -> None:
        for attr in dir(table_column):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(table_column, attr))
        self.table_column = table_column
        self.index_column = index_column
        self.root_path = root_path

    def get_column(self):
        return self.table_column

    def get_key(self):
        key_type = ""
        if self.table_column.is_primary():
            key_type = " class='primaryKey' title='Primary Key'"
        elif self.table_column.is_foreign_key():
            key_type = " class='foreignKey' title='Foreign Key'"
        elif self.index_column:
            key_type = " class='" + self.mark_as_index_column() + "' title='Indexed'"

        return key_type

    def get_key_title(self):
        key_title = ""
        if self.table_column.is_primary():
            key_title = "Primary Key"
        elif self.table_column.is_foreign_key():
            key_title = "Foreign Key"
        elif self.index_column:
            key_title = "Indexed"

        return key_title

    def get_key_class(self):
        key_class = ""
        if self.table_column.is_primary():
            key_class = "primaryKey"
        elif self.table_column.is_foreign_key():
            key_class = "foreignKey"
        elif self.index_column:
            key_class = "indexedColumn"

        return key_class

    def get_key_icon(self):
        key_icon = ""
        if self.table_column.is_primary() or self.table_column.is_foreign_key():
            key_icon = "<i class='icon ion-key iconkey' style='padding-left: 5px;'></i>"
        elif self.index_column:
            key_icon = "<i class='fa fa-sitemap fa-rotate-120' style='padding-right: 5px;'></i>"

        return key_icon

    def get_nullable(self):
        if not self.table_column.not_null:
            return "√"
        else:
            return ""

    def get_title_nullable(self):
        if self.table_column.not_null:
            return "nullable"
        else:
            return ""

    def get_auto_updated(self):
        if self.table_column.is_auto_updated:
            return "√"
        else:
            return ""

    def get_title_auto_updated(self):
        if self.table_column.is_auto_updated:
            return "Automatically updated by the database"
        else:
            return ""

    def get_default_value(self):
        return str(self.table_column.get_default_value())

    def mark_as_index_column(self):
        if self.index_column:
            return "indexedColumn"
        return ""
