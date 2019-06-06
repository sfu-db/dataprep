"""
dataprep module
"""
import logging

# Dask Default partitions
DEFAULT_PARTITIONS = 1

logging.basicConfig(level=logging.INFO, format="%(message)")
LOGGER = logging.getLogger(__name__)
