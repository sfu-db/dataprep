from .pagedata import PageData
from .pystache_constraints import PSConstraints


class HtmlConstraintsPage:
    def __init__(self, pystache_object) -> None:
        self.pystache_object = pystache_object

    def page_writer(self, constraints, table, new_file):
        page_data = PageData("constraint.html", "constraint.js")
        page_data.addScope("constraints", constraints)
        page_data.addScope("constraints_num", len(constraints))
        page_data.addScope("checkConstraints", HtmlConstraintsPage.collect_check_constraints(table))
        page_data.setDepth(0)
        pagination_configs = {
            "fkTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "checkTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            page_data, new_file, "constraint.js", pagination_configs
        )

    def collect_check_constraints(tables):
        all_constraints = []
        results = []
        for table in tables:
            if len(table.getCheckConstraints()) > 0:
                all_constraints.append(table.getCheckConstraints())
        for x in all_constraints:
            results.append(PSConstraints(x, x.keys(), x.values()))

    """
    def collect_check_constraints(table):
        all_constraints=[]
        results=[]
        if len(table.getCheckConstraints())>0:
                all_constraints.append(table.getCheckConstraints())
        for x in all_constraints:
            results.append(PC.ps_constraints(x,x.keys(),x.values()))
        return results
    """
