from .db_meta import DbMeta


class Database:
    def __init__(self, name: str, schema: str, stats: DbMeta) -> None:
        self.name = name.replace("'", "")
        self.schema = schema
        self.tables = {}
        self.views = {}
        for key, value in stats.__dict__.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def add_table(self, table_name: str, table_object):
        self.tables[table_name] = table_object

    def add_view(self, view_name: str, view_object):
        self.views[view_name] = view_object

    def get_tables(self):
        return self.tables.values()

    def get_tables_dict(self):
        return self.tables

    def get_views(self):
        return self.views.values()

    def get_name(self):
        return self.name
