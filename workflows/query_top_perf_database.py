import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import DictCursor
import logging
from collections import defaultdict
import json
from workflows.utils import get_repo_root_path

logger = logging.getLogger("run_log")


def build_query(model_name: str, device: str) -> Tuple[str, Tuple]:
    """
    Build SQL query and parameters based on whether device is 'all' or specific device.

    Args:
        model_name: The model name to filter by
        device: Either a specific device name or 'all' for all devices

    Returns:
        Tuple of (query_string, parameters_tuple)
    """
    base_query = 'SELECT * FROM "sw_test"."customer_benchmark_target" WHERE customer_benchmark_target.model_name = %s'

    if device.lower() == "all":
        query = base_query
        params = (model_name,)
        logger.info(
            f"Executing query with parameters: model_name='{model_name}', device='ALL'"
        )
    else:
        query = base_query + " AND customer_benchmark_target.device = %s"
        params = (model_name, device)
        logger.info(
            f"Executing query with parameters: model_name='{model_name}', device='{device}'"
        )

    return query, params


def create_db_connection() -> Optional[psycopg2.extensions.connection]:
    """
    Create and return a database connection.

    Returns:
        Database connection object or None if connection fails
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv("PERF_DB_HOST", None),
            port=os.getenv("PERF_DB_PORT", None),
            database=os.getenv("PERF_DB_NAME", None),
            user=os.getenv("PERF_DB_USER", None),
            password=os.getenv("PERF_DB_PASSWORD", None),
        )
        return connection
    except psycopg2.OperationalError:
        logger.error("Error: Could not connect to the database.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while connecting: {e}")
        return None


def fetch_data(model_name: str, device: str) -> Optional[List[Dict[str, Any]]]:
    """
    Main function to fetch data from the database.

    Args:
        model_name: The model name to search for
        device: Either a specific device name or 'all' for all devices

    Returns:
        List of records as dictionaries, or None if operation fails
    """
    # Build query based on device parameter
    query, params = build_query(model_name, device)

    # Create database connection
    connection = create_db_connection()
    if connection is None:
        return None

    cursor = None
    try:
        cursor = connection.cursor(cursor_factory=DictCursor)
        cursor.execute(query, params)
        records = cursor.fetchall()

        if not records:
            logger.info(
                f"No records found for model '{model_name}' and device '{device}'"
            )
            return None

        return records

    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return None

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


def organize_records(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Organize records by device.

    Args:
        records: List of database records

    Returns:
        Dict mapping device names to lists of benchmark configurations
    """
    organized_records = defaultdict(lambda: defaultdict(list))
    for record in records:
        device = device_rename(record["device"])
        organized_records[record["model_name"]][device].append(
            {
                "isl": record["input_sequence_length"],
                "osl": record["input_sequence_length"],
                "max_concurrency": 1,
                "num_prompts": 8,
                "task_type": "image" if record["image_height"] is not None else "text",
                "image_height": record["image_height"],
                "image_width": record["image_width"],
                "images_per_prompt": 1 if record["image_height"] is not None else None,
                "targets": {
                    "theoretical": {
                        "ttft_ms": record["ttft_comms_ms"],
                        "tput_user": record["t_s_u"],
                        "tput": record["throughput_t_s"],
                    }
                },
            }
        )

    return organized_records


def device_rename(device: str) -> str:
    """
    Rename device to match DeviceTypes.
    """
    mapping = {
        "WH_T3K": "t3k",
        "WH_Galaxy": "galaxy",
    }
    return mapping.get(device, device)


def top_perf_database_records(
    model_name: str, device: str = "all"
) -> Optional[Dict[str, List[Dict[str, Any]]]]:
    """
    Query the database and organize the records.

    Args:
        model_name: The model name to search for
        device: Either a specific device name or 'all' for all devices

    Returns:
        Dict mapping device names to lists of benchmark configurations, or None if query fails
    """
    records = fetch_data(model_name, device)
    if records is None:
        return None
    return organize_records(records)


def fetch_all_models() -> Dict[str, List[Dict[str, Any]]]:
    query = "SELECT * FROM sw_test.customer_benchmark_target"

    try:
        connection = create_db_connection()
        cursor = connection.cursor(cursor_factory=DictCursor)
        cursor.execute(query)
        records = cursor.fetchall()
        organized_records = organize_records(records)
        return organized_records

    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        return {}

    finally:
        if cursor is not None:
            cursor.close()
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    organized_records = fetch_all_models()
    json_path = (
        get_repo_root_path()
        / "benchmarking"
        / "benchmark_targets"
        / "model_performance_reference.json"
    )
    with open(json_path, "w") as f:
        json.dump(organized_records, f)

    print(f"Saved organized records to {json_path}")
