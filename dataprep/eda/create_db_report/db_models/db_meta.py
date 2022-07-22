class DbMeta:
    num_of_views = 0
    num_of_schemas = 0
    num_of_fk = 0
    num_of_uk = 0
    num_of_pk = 0
    num_of_tables = 0

    engine_name_dict = {
        "mysql": "MySQL",
        "postgresql": "PostgreSQL",
        "sqlite": "SQLite",
    }

    def __init__(
        self,
        engine_name: str,
        num_of_views: int,
        num_of_schemas: int,
        num_of_fk: int,
        num_of_uk: int,
        num_of_pk: int,
        num_of_tables: int,
        product_version: str,
        connection_url: str,
    ) -> None:
        self.num_of_views = num_of_views
        self.num_of_schemas = num_of_schemas
        self.num_of_fk = num_of_fk
        self.num_of_uk = num_of_uk
        self.num_of_pk = num_of_pk
        self.num_of_table = num_of_tables
        self.connection_url = connection_url
        self.database_product = self.engine_name_dict[engine_name] + " - " + product_version
