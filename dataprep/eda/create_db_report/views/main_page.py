from datetime import datetime
from tokenize import String
from .pagedata import PageData
from .pystache_tables import PSTable
from .template_pystache import Template
from ..model.add_foreignkey import ForeignKeyConstraint
from ..model.db_metadata import DbMeta
from ..model.init_database import Database


class HtmlMainIndexPage:

    def __init__(self, pystache_object, description, stats) -> None:
        self.pystache_object = pystache_object
        self.description = description
        self.stats = stats

    def page_writer(self, database: Database, tables, view, implied_constraints, new_file):
        columns_amount = 0
        tables_amount = 0
        views_amount = 0
        constraints_amount = len(
            ForeignKeyConstraint.getAllForeignKeyConstraints(database.getTables())
        )
        routines_amount = 0
        anomalies_amount = 0
        generation_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        all_tables = []
        for t in tables:
            if t.is_view():
                views_amount += 1
            else:
                tables_amount += 1
            columns_amount += len(t.getColumns())
            all_tables.append(PSTable(t, ""))

        page_data = PageData("main.html", "main.js")
        page_data.addScope("database_name", database.getName())
        page_data.addScope("generation_time", generation_time)
        page_data.addScope("tablesAmount", tables_amount)
        page_data.addScope("viewsAmount", views_amount)
        page_data.addScope("columnsAmount", columns_amount)
        page_data.addScope("constraintsAmount", constraints_amount)
        page_data.addScope("routinesAmount", routines_amount)
        page_data.addScope("anomaliesAmount", anomalies_amount)
        page_data.addScope("tables", all_tables)
        page_data.addScope("database", database)
        page_data.addScope("schema", database.getSchema())
        page_data.setDepth(0)

        pagination_configs = {
            "databaseObjects": {"paging": "true", "pageLength": 10, "lengthChange": "false"}
        }
        return self.pystache_object.write_data(page_data, new_file, "main.js", pagination_configs)
