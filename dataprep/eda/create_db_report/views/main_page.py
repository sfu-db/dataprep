import views.pagedata as PD
import views.pystache_tables
import model.add_foreignkey as FK
import views.template_pystache as template_pystache
from tokenize import String
from model.db_metadata import DbMeta
from model.init_database import Database
from datetime import datetime


class HtmlMainIndexPage:
    pystache_object: template_pystache
    description: String
    stats: DbMeta

    def __init__(self, pystache_object, description, stats) -> None:
        self.pystache_object = pystache_object
        self.description = description
        self.stats = stats

    def pageWriter(self, database: Database, tables, view, impliedConstraints, new_file):

        columnsAmount = 0
        tablesAmount = 0
        viewsAmount = 0
        constraintsAmount = len(
            FK.ForeignKeyConstraint.getAllForeignKeyConstraints(database.getTables())
        )
        routinesAmount = 0
        anomaliesAmount = 0
        generationTime = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        allTables = []
        for t in tables:
            if t.is_view():
                viewsAmount += 1
            else:
                tablesAmount += 1
            columnsAmount += len(t.getColumns())
            allTables.append(views.pystache_tables.ps_table(t, ""))

        pageData = PD.pageData("main.html", "main.js")
        pageData.addScope("database_name", database.getName())
        pageData.addScope("generation_time", generationTime)
        pageData.addScope("tablesAmount", tablesAmount)
        pageData.addScope("viewsAmount", viewsAmount)
        pageData.addScope("columnsAmount", columnsAmount)
        pageData.addScope("constraintsAmount", constraintsAmount)
        pageData.addScope("routinesAmount", routinesAmount)
        pageData.addScope("anomaliesAmount", anomaliesAmount)
        pageData.addScope("tables", allTables)
        pageData.addScope("database", database)
        pageData.addScope("schema", database.getSchema())
        pageData.setDepth(0)

        pagination_configs = {
            "databaseObjects": {"paging": "true", "pageLength": 10, "lengthChange": "false"}
        }
        return self.pystache_object.write_data(pageData, new_file, "main.js", pagination_configs)
