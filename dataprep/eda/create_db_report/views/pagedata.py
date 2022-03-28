from tokenize import String


class PageData:
    templateName: String
    scope = {}
    scriptName: String
    depth = 0

    def __init__(self, templateName: String, scriptName: String):
        self.templateName = templateName
        self.scriptName = scriptName

    def getTemplateName(self):
        return self.templateName

    def getScope(self):
        return self.scope

    def getScriptName(self):
        return self.scriptName

    def getDepth(self):
        return self.depth

    def addScope(self, key, value):
        self.scope[key] = value

    def setDepth(self, depth):
        self.depth = depth

    def setScriptName(self, scriptName):
        self.scriptName = scriptName
