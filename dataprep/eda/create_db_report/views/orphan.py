import json
from typing import List, Any
from ..page_models.page_data import PageData
from ..page_models.page_template import PageTemplate


class OrphanPage:
    def __init__(self, template_object: PageTemplate) -> None:
        self.template_object = template_object

    def page_writer(
        self,
        json_tables: List[Any],
        json_relationships: List[Any],
        new_file: str,
    ):
        """
        Compile the data needed by the pystache template for orphan page
        """
        page_data = PageData("orphan.html", "")
        page_data.add_scope("diagram_tables", json.dumps(json_tables))
        page_data.add_scope("diagram_relationships", json.dumps(json_relationships))
        page_data.set_depth(0)

        return self.template_object.write_data(page_data, new_file, "", {})
