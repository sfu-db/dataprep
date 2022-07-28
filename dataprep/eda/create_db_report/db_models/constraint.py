from .table import Table


class ForeignKeyConstraint:
    imported_key_cascade = "0"
    imported_key_restrict = "1"
    imported_key_set_null = "2"
    imported_key_no_action = "3"

    def __init__(self, child: Table, name: str, delete_rule: str, update_rule: str):
        self.name = name.replace("'", "")
        self.delete_rule = delete_rule
        self.update_rule = update_rule
        self.parent_columns = []
        self.child_columns = []
        self.parent_table = None
        self.child_table = child

    def add_parent_column(self, column):
        if column is not None:
            self.parent_columns.append(column)
            self.parent_table = column.table

    def add_child_column(self, column):
        if column is not None:
            self.child_columns.append(column)

    def get_parent_table(self):
        return self.parent_table

    def get_child_table(self):
        return self.child_table

    def is_cascade_on_delete(self):
        return self.delete_rule == self.imported_key_cascade

    def is_restrict_delete(self):
        return (
            self.delete_rule == self.imported_key_no_action
            or self.delete_rule == self.imported_key_restrict
        )

    def is_null_on_delete(self):
        return self.delete_rule == self.imported_key_set_null

    def get_delete_rule_name(self):
        if self.delete_rule == self.imported_key_cascade:
            return "Cascade on delete"
        elif (
            self.delete_rule == self.imported_key_restrict
            or self.delete_rule == self.imported_key_no_action
        ):
            return "Restrict delete"
        elif self.delete_rule == self.imported_key_set_null:
            return "Null on delete"
        else:
            return ""

    def get_delete_rule_description(self):
        if self.delete_rule == self.imported_key_cascade:
            return "Cascade on delete:\nDeletion of parent deletes child"
        elif (
            self.delete_rule == self.imported_key_restrict
            or self.delete_rule == self.imported_key_no_action
        ):
            return "Restrict delete:\nParent cannot be deleted if children exist"
        elif self.delete_rule == self.imported_key_set_null:
            return "Null on delete:\nForeign key to parent set to NULL when parent deleted"
        else:
            return ""

    def get_delete_rule_alias(self):
        if self.delete_rule == self.imported_key_cascade:
            return "C"
        elif (
            self.delete_rule == self.imported_key_restrict
            or self.delete_rule == self.imported_key_no_action
        ):
            return "R"
        elif self.delete_rule == self.imported_key_set_null:
            return "N"
        else:
            return ""

    @staticmethod
    def get_all_foreign_key_constraints(tables):
        constraints = []

        for table in tables:
            constraints.extend(table.get_foreign_keys())

        return constraints
