import os
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import DictCursor
import logging

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
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
        )
        return connection
    except psycopg2.OperationalError:
        raise Exception("Error: Could not connect to the database.")
    except Exception as e:
        raise Exception(f"An error occurred while connecting: {e}")


def format_and_print_results(
    records: List[Dict[str, Any]], model_name: str, device: str
) -> None:
    """
    Format and print the query results in a readable way.

    Args:
        records: List of record dictionaries from the database
        model_name: The model name that was queried
        device: The device that was queried (or 'all')
    """
    print("\n--- Query Results ---")
    print(
        f"Found {len(records)} records for model '{model_name}' and device '{device}'\n"
    )

    if not records:
        raise Exception("No records found.")

    # Print each row with dictionary-style access
    for i, row in enumerate(records, 1):
        print(f"Record {i}:")
        for col_name, value in row.items():
            print(f"  {col_name}: {value}")
        print()  # Empty line between records

    # Show example of direct column access and available columns
    print("--- Example of direct column access ---")
    first_record = records[0]
    print(f"Model Name: {first_record['model_name']}")
    print(f"Device: {first_record['device']}")
    print(f"Available columns: {list(first_record.keys())}")


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

    cursor = None
    try:
        cursor = connection.cursor(cursor_factory=DictCursor)
        cursor.execute(query, params)
        records = cursor.fetchall()

    except Exception as e:
        raise Exception(f"Failed to execute query: {e}")

    finally:
        if cursor is not None:
            cursor.close()
        connection.close()

    return records


if __name__ == "__main__":
    print("=== Example 1: Specific Device ===")
    records = fetch_data("Llama-3.1-8B-Instruct", "n300")
    for record in records:
        for key, value in record.items():
            print(f"{key}: {value}")
        print()

    print("\n" + "=" * 50)
    print("=== Example 2: All Devices ===")
    records = fetch_data("Llama-3.1-8B-Instruct", "all")
    for record in records:
        for key, value in record.items():
            print(f"{key}: {value}")
        print()
