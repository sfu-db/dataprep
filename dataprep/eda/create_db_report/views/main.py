from typing import List
from datetime import datetime
from ..db_models.database import Database
from ..db_models.db_meta import DbMeta
from ..db_models.constraint import ForeignKeyConstraint
from ..db_models.table import Table
from ..page_models.page_data import PageData
from ..page_models.page_template import PageTemplate
from ..template_models.table import TemplateTable


class MainPage:
    def __init__(self, template_object: PageTemplate, description: str, stats: DbMeta) -> None:
        self.template_object = template_object
        self.description = description
        self.stats = stats

    def page_writer(self, database: Database, tables: List[Table], new_file: str):
        """
        Compile the data needed by the template for index page
        """
        columns_amount = 0
        tables_amount = 0
        views_amount = 0
        constraints_amount = len(
            ForeignKeyConstraint.get_all_foreign_key_constraints(database.get_tables())
        )
        routines_amount = 0
        anomalies_amount = 0
        generation_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        all_tables = []
        for table in tables:
            if table.is_view():
                views_amount += 1
            else:
                tables_amount += 1
            columns_amount += len(table.get_columns())
            all_tables.append(TemplateTable(table))

        page_data = PageData("main.html", "main.js")
        page_data.add_scope("database_name", database.name)
        page_data.add_scope("generation_time", generation_time)
        page_data.add_scope("tables_amount", tables_amount)
        page_data.add_scope("views_amount", views_amount)
        page_data.add_scope("columns_amount", columns_amount)
        page_data.add_scope("constraints_amount", constraints_amount)
        page_data.add_scope("routines_amount", routines_amount)
        page_data.add_scope("anomalies_amount", anomalies_amount)
        page_data.add_scope("tables", all_tables)
        page_data.add_scope("database", database)
        page_data.add_scope("schema", database.schema)
        page_data.set_depth(0)

        pagination_configs = {
            "database_objects": {"paging": "true", "pageLength": 10, "lengthChange": "false"}
        }
        return self.template_object.write_data(page_data, new_file, "main.js", pagination_configs)
