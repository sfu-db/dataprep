import webbrowser


class Report:
    """
    This class creates a customized Report object for the create_db_report function
    """

    def __init__(self, report: str, path: str) -> None:
        self.report = report
        self.path = path

    def show_browser(self) -> None:
        """
        Open the report in the browser. This is useful when calling from terminal.
        """

        with open(self.path, "w", encoding="utf-8") as file:
            file.write(self.report)
        webbrowser.open(f"file://{self.path}", new=2)
