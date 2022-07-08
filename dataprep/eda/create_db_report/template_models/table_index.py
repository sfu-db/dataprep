from ..db_models.table_index import TableIndex


class TemplateTableIndex:
    def __init__(self, index: TableIndex) -> None:
        for attr in dir(index):
            if not attr.startswith("__"):
                setattr(self, attr, getattr(index, attr))
        self.index = index

    def get_index(self):
        return self.index

    def get_key(self):
        if self.index.is_primary:
            key_type = " class='primaryKey' title='Primary Key'"
        elif self.index.is_unique:
            key_type = " class='uniqueKey' title='Unique Key'"
        else:
            key_type = " title='Indexed'"
        return key_type

    def get_key_icon(self):
        key_icon = ""
        if self.index.is_primary or self.index.is_unique:
            key_icon = "<i class='icon ion-key iconkey'></i> "

        return key_icon
