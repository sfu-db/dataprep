from .pagedata import PageData
from .pystache_constraints import PSConstraints
from .template_pystache import Template
from ..model.add_foreignkey import ForeignKeyConstraint
from ..model.init_table import Table
from typing import List


class HtmlConstraintsPage:
    def __init__(self, pystache_object: Template) -> None:
        self.pystache_object = pystache_object

    def page_writer(
        self, constraints: List[ForeignKeyConstraint], tables: List[Table], new_file: str
    ):
        """
        Compile the data needed by the pystache template for constraints page
        """
        page_data = PageData("constraint.html", "constraint.js")
        page_data.addScope("constraints", constraints)
        page_data.addScope("constraints_num", len(constraints))
        page_data.addScope("checkConstraints", self.collect_check_constraints(tables))
        page_data.setDepth(0)
        pagination_configs = {
            "fkTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "checkTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            page_data, new_file, "constraint.js", pagination_configs
        )

    @staticmethod
    def collect_check_constraints(tables: List[Table]):
        all_constraints = []
        results = []
        for table in tables:
            if len(table.getCheckConstraints()) > 0:
                all_constraints.append(table.getCheckConstraints())
        for x in all_constraints:
            results.append(PSConstraints(x, x.keys(), x.values()))
