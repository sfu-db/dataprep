from typing import Any


class PageData:
    def __init__(self, template_name: str, script_name: str):
        self.template_name = template_name
        self.script_name = script_name
        self.scope = {}
        self.depth = 0

    def add_scope(self, key: str, value: Any):
        self.scope[key] = value

    def set_depth(self, depth: int):
        self.depth = depth
