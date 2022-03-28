import views.pagedata as PD
import views.pystache_constraints as PC


class HtmlConstraintsPage:
    def __init__(self, pystache_object) -> None:
        self.pystache_object = pystache_object

    def pageWriter(self, constraints, table, new_file):
        pagedata = PD.pageData("constraint.html", "constraint.js")
        pagedata.addScope("constraints", constraints)
        pagedata.addScope("constraints_num", len(constraints))
        pagedata.addScope("checkConstraints", HtmlConstraintsPage.collectCheckConstraints(table))
        pagedata.setDepth(0)
        pagination_configs = {
            "fkTable": {"paging": "true", "pageLength": 20, "lengthChange": "false"},
            "checkTable": {"paging": "true", "pageLength": 10, "lengthChange": "false"},
        }
        return self.pystache_object.write_data(
            pagedata, new_file, "constraint.js", pagination_configs
        )

    def collectCheckConstraints(tables):
        all_constraints = []
        results = []
        for table in tables:
            if len(table.getCheckConstraints()) > 0:
                all_constraints.append(table.getCheckConstraints())
        for x in all_constraints:
            results.append(PC.ps_constraints(x, x.keys(), x.values()))

    """
    def collectCheckConstraints(table):
        all_constraints=[]
        results=[]
        if len(table.getCheckConstraints())>0:
                all_constraints.append(table.getCheckConstraints())
        for x in all_constraints:
            results.append(PC.ps_constraints(x,x.keys(),x.values()))
        return results
    """
