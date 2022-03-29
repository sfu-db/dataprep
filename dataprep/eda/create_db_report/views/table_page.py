from .pagedata import PageData
from .pystache_columns import MustacheTableColumn
from .pystache_index import PSIndex
from .template_pystache import Template

class HtmlTablePage:
    def __init__(self, pystache_object:Template) -> None:
        self.pystache_object = pystache_object

    "compiles the data needed by the pystache template for tables pages"
    
    def page_writer(self, table, new_file):
        primaries = set(table.getPrimaryColumns())
        indexes = set()
        table_columns = set()

        for i in table.getIndexes():
            indexes.add(PSIndex(i))

        for c in table.getColumns():
            table_columns.add(MustacheTableColumn(c, None, ""))
        check_constraints = None  # HtmlTablePage.collect_check_constraints(table)

        page_data = PageData("tables/table.html", "table.js")
        page_data.addScope("table", table)
        page_data.addScope("primaries", primaries)
        page_data.addScope("columns", table_columns)
        page_data.addScope("indexes", indexes)
        page_data.addScope("checkConstraints", check_constraints)
        page_data.addScope("sqlCode", self.sqlCode(table))
        page_data.setDepth(0)

        pagination_configs = {
            "standardTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "indexesTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
            "checkTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            page_data, new_file, "table.js", pagination_configs, "../"
        )

    def sqlCode(self, table):
        if table.getViewDefinition() != None:
            return table.getViewDefinition().strip()
        return ""
