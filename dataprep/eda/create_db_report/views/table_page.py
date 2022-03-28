import json
import views.pagedata as PD
import views.pystache_columns as PC
import views.pystache_index as PI


class HtmlTablePage:
    def __init__(self, pystache_object) -> None:
        self.pystache_object = pystache_object

    def pageWriter(self, table, new_file):
        primaries = set(table.getPrimaryColumns())
        indexes = set()
        table_columns = set()

        for i in table.getIndexes():
            indexes.add(PI.ps_index(i))

        for c in table.getColumns():
            table_columns.add(PC.MustacheTableColumn(c, None, ""))
        check_constraints = None  # HtmlTablePage.collectCheckConstraints(table)

        pageData = PD.pageData("tables/table.html", "table.js")
        pageData.addScope("table", table)
        pageData.addScope("primaries", primaries)
        pageData.addScope("columns", table_columns)
        pageData.addScope("indexes", indexes)
        pageData.addScope("checkConstraints", check_constraints)
        pageData.addScope("sqlCode", self.sqlCode(table))
        pageData.setDepth(0)

        pagination_configs = {
            "standardTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "indexesTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
            "checkTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            pageData, new_file, "table.js", pagination_configs, "../"
        )

    def sqlCode(self, table):
        if table.getViewDefinition() != None:
            return table.getViewDefinition().strip()
        return ""
