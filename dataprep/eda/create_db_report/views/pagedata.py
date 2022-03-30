from typing import Any


class PageData:
    def __init__(self, templateName: str, scriptName: str):
        self.templateName = templateName
        self.scriptName = scriptName
        self.scope = {}
        self.depth = 0

    def getTemplateName(self):
        return self.templateName

    def getScope(self):
        return self.scope

    def getScriptName(self):
        return self.scriptName

    def getDepth(self):
        return self.depth

    def addScope(self, key: str, value: Any):
        self.scope[key] = value

    def setDepth(self, depth: int):
        self.depth = depth

    def setScriptName(self, scriptName: str):
        self.scriptName = scriptName
