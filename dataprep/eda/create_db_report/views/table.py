import json
from typing import List, Any
from ..db_models.table import Table
from ..page_models.page_data import PageData
from ..page_models.page_template import PageTemplate
from ..template_models.table_column import TemplateTableColumn
from ..template_models.table_index import TemplateTableIndex


class TablePage:
    def __init__(self, template_object: PageTemplate) -> None:
        self.template_object = template_object

    def page_writer(
        self, table: Table, json_tables: List[Any], json_relationships: List[Any], new_file: str
    ):
        """
        Compile the data needed by the template for tables pages
        """
        primaries = set(table.primary_keys)
        indexes = set()
        table_columns = set()

        for i in table.get_indexes():
            indexes.add(TemplateTableIndex(i))

        for c in table.get_columns():
            table_columns.add(TemplateTableColumn(c, False, ""))
        check_constraints = None  # HtmlTablePage.collect_check_constraints(table)

        page_data = PageData("tables/table.html", "table.js")
        page_data.add_scope("table", table)
        page_data.add_scope("primaries", primaries)
        page_data.add_scope("columns", table_columns)
        page_data.add_scope("indexes", indexes)
        page_data.add_scope("check_constraints", check_constraints)
        page_data.add_scope("sql_code", self.sql_code(table))
        page_data.add_scope("diagram_tables", json.dumps(json_tables))
        page_data.add_scope("diagram_relationships", json.dumps(json_relationships))
        page_data.set_depth(0)

        pagination_configs = {
            "standard_table": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "indexes_table": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
            "check_table": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.template_object.write_data(
            page_data, new_file, "table.js", pagination_configs, "../"
        )

    @staticmethod
    def sql_code(table: Table):
        if table.get_view_definition() is not None:
            return table.get_view_definition().strip()
        return ""
