"""Defines ConfigState."""

from typing import Optional

from dataprep.connector.schema.defs import ConfigDef


class ConfigState:
    """ConfigState"""

    config: Optional[ConfigDef] = None

    def __init__(self, config: Optional[ConfigDef]) -> None:
        self.config = config

    def __add__(self, rhs: ConfigDef) -> "ConfigState":
        if self.config is None:
            return ConfigState(rhs)

        return ConfigState(self.config.merge(rhs))

    def __str__(self) -> str:
        return str(self.config)

    def __repr__(self) -> str:
        return str(self)
