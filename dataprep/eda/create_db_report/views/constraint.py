from typing import List
from ..db_models.constraint import ForeignKeyConstraint
from ..db_models.table import Table
from ..page_models.page_data import PageData
from ..page_models.page_template import PageTemplate
from ..pystache_models.pystache_constraint import PystacheConstraint


class ConstraintPage:
    def __init__(self, pystache_object: PageTemplate) -> None:
        self.pystache_object = pystache_object

    def page_writer(
        self, constraints: List[ForeignKeyConstraint], tables: List[Table], new_file: str
    ):
        """
        Compile the data needed by the pystache template for constraints page
        """
        page_data = PageData("constraint.html", "constraint.js")
        page_data.add_scope("constraints", constraints)
        page_data.add_scope("constraints_num", len(constraints))
        page_data.add_scope("check_constraints", self.collect_check_constraints(tables))
        page_data.set_depth(0)
        pagination_configs = {
            "fk_table": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "check_table": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            page_data, new_file, "constraint.js", pagination_configs
        )

    @staticmethod
    def collect_check_constraints(tables: List[Table]):
        all_constraints = []
        results = []
        for table in tables:
            if len(table.check_constraints) > 0:
                all_constraints.append(table.check_constraints)
        for x in all_constraints:
            results.append(PystacheConstraint(x, x.keys(), x.values()))