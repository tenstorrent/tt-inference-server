from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
import json
import pandas as pd
from pathlib import Path
import socket
from report_config import BenchmarkMeasurement, CompleteBenchmarkRun
import re
from io import StringIO
import paramiko
import uuid
from typing import List, Optional
import tempfile
import os

########################################################################################
# Data Pipeline Upload
########################################################################################

def upload_superset(
    benchmark_runs: List[CompleteBenchmarkRun],
    sftp_endpoint: str = "benchmark-writer@s-dbd4b8a190fa40a4b.server.transfer.us-east-2.amazonaws.com",
) -> List[str]:
    """
    Upload benchmark run data to the data pipeline via SFTP.

    Args:
        benchmark_runs (List[CompleteBenchmarkRun]): List of benchmark runs to upload.
        sftp_endpoint (str): SFTP endpoint in format "user@host".
        target_bucket (str): Target S3 bucket path for upload.
        ssh_key_path (Optional[str]): Path to SSH private key file.
        ssh_password (Optional[str]): SSH password (if not using key-based auth).

    Returns:
        List[str]: List of uploaded file paths.
    """
    uploaded_files = []
    
    # Parse SFTP endpoint
    if "@" not in sftp_endpoint:
        raise ValueError(f"Invalid SFTP endpoint format: {sftp_endpoint}")
    
    username, hostname = sftp_endpoint.split("@", 1)
    
    # Setup SSH client
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:

        ssh_client.connect(hostname, 
                           username=username,
                           allow_agent=True,
                           look_for_keys=True)
        
        # Open SFTP session
        sftp_client = ssh_client.open_sftp()
        
        # Upload each benchmark run as a separate JSON file
        for i, benchmark_run in enumerate(benchmark_runs):
            # Generate unique filename
            timestamp = benchmark_run.run_start_ts.strftime("%Y%m%d_%H%M%S")
            run_id = str(uuid.uuid4())[:8]
            filename = f"benchmark_test_run_{timestamp}_{run_id}.json"
            
            # Convert to JSON
            benchmark_data = benchmark_run.model_dump(mode='json')
            json_content = json.dumps(benchmark_data, indent=2, default=str)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_file.write(json_content)
                temp_file_path = temp_file.name
            
            try:
                # Upload to SFTP
                remote_path = f"{filename}"
                sftp_client.put(temp_file_path, remote_path)
                uploaded_files.append(remote_path)
                print(f"Successfully uploaded: {remote_path}")
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
        
        sftp_client.close()
        
    finally:
        ssh_client.close()
    
    return uploaded_files

def upload_benchmark_pipeline(
    benchmark_dir: str, 
    monitor_file: str, 
) -> List[str]:
    """
    Complete pipeline to process and upload benchmark data.
    
    Args:
        benchmark_dir (str): Path to the benchmark directory.
        monitor_file (str): Path to the monitor file.
        ssh_key_path (Optional[str]): Path to SSH private key file.
        ssh_password (Optional[str]): SSH password (if not using key-based auth).
    
    Returns:
        List[str]: List of uploaded file paths.
    """
    # Process benchmark data
    benchmark_runs = upload_benchmark_report(benchmark_dir, monitor_file)
    
    # Upload to data pipeline
    uploaded_files = upload_superset(
        benchmark_runs=benchmark_runs,
    )
        return uploaded_files

########################################################################################
# Benchmark
########################################################################################

def upload_benchmark_report(benchmark_dir: str, monitor_file: str) -> List[CompleteBenchmarkRun]:
    """Upload benchmark report to the database.

    Args:
        benchmark_dir (str): Path to the benchmark directory.
        monitor_file (str): Path to the monitor file,which includes the power and temperature, memory usage, etc.

    Returns:
        List[CompleteBenchmarkRun]: List of complete benchmark run objects.
    """
    benchmark_list_files = Path(benchmark_dir).glob("benchmark_id_*.json")
    monitor_df = pd.read_json(monitor_file, lines=True)
    all_measurements = []
    benchmark_run = []
    for benchmark_file in benchmark_list_files:
        with open(benchmark_file) as report_file:
            report_data = json.load(report_file)
        
        start_time_utc, end_time_utc = parse_time_to_utc(report_data["date"], report_data["duration"])
        filtered_report = filter_benchmark_report(report_data,
                                                    ["mean_ttft_ms",
                                                    "std_ttft_ms",
                                                    "mean_tpot_ms",
                                                    "std_tpot_ms",
                                                    "mean_tps",
                                                    "std_tps",
                                                    "tps_decode_throughput",
                                                    "tps_prefill_throughput",
                                                    "mean_e2el_ms",
                                                    "request_throughput"])
                                        
        device_power_avg = None
        device_temperature_avg = None
        
        if not monitor_df.empty:
            device_power_avg, device_temperature_avg = calculate_average_device_metrics(
                monitor_df, start_time_utc, end_time_utc
            )
        
        start_dt = datetime.fromisoformat(start_time_utc.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(end_time_utc.replace('Z', '+00:00'))
        
        for metric_name, metric_value in filtered_report.items():
            measurement = BenchmarkMeasurement(
                step_start_ts=start_dt,
                step_end_ts=end_dt,
                iteration=1,
                step_name="benchmark",
                name=metric_name,
                value=float(metric_value),
                device_power=device_power_avg,
                device_temperature=device_temperature_avg,

            )
            all_measurements.append(measurement)
    
        if all_measurements:
            run_start_ts = min(measurement.step_start_ts for measurement in all_measurements)
            run_end_ts = max(measurement.step_end_ts for measurement in all_measurements)
        else:
            
            run_start_ts = datetime.now(timezone.utc)
            run_end_ts = run_start_ts
    
        device_info = create_device_info_dict(monitor_df) if not monitor_df.empty else None
    
        benchmark_run.append(CompleteBenchmarkRun(
            run_start_ts=run_start_ts,
            run_end_ts=run_end_ts,
            run_type="competitor_benchmark",
            device_hostname=socket.gethostname(),
            ml_model_name=report_data["model_id"], 
            device_info=device_info,
            measurements=all_measurements,
            input_sequence_length=report_data["input_lens"][0],
            output_sequence_length=report_data["output_lens"][0],
            batch_size=report_data["max_concurrency"],
            training=False
        ))
        all_measurements = []
    
    return benchmark_run


def calculate_average_device_metrics(monitor_df: pd.DataFrame, start_time_utc: str, end_time_utc: str) -> tuple[float, float]:
    """
    Calculate average power and temperature across all devices for the time period.

    Args:
        monitor_df (pd.DataFrame): Monitor data frame.
        start_time_utc (str): Start time in UTC.
        end_time_utc (str): End time in UTC.

    Returns:
        tuple[float, float]: Average power and temperature.
    """
    start_dt = datetime.fromisoformat(start_time_utc.replace('Z', '+00:00'))
    end_dt = datetime.fromisoformat(end_time_utc.replace('Z', '+00:00'))
    
    if monitor_df["timestamp"].dtype == 'object':
        monitor_df["timestamp"] = pd.to_datetime(monitor_df["timestamp"], utc=True)
    
    filtered_df = monitor_df[(monitor_df["timestamp"] >= start_dt) & (monitor_df["timestamp"] <= end_dt)]
    
    if filtered_df.empty:
        print("Warning: No data found in the specified time range")
        return 0.0, 0.0
    
    mean_power = float(filtered_df["power_watts"].mean())
    mean_temperature = float(filtered_df["temperature"].mean())
    
    return mean_power, mean_temperature


def create_device_info_dict(monitor_df: pd.DataFrame) -> dict:
    """Create device info dictionary from monitor data.

    Args:
        monitor_df (pd.DataFrame): Monitor data frame.

    Returns:
        dict: includes device name, gpu id, etc.
    """
    if monitor_df.empty:
        return {}
    
    device_info = {}
    for idx in monitor_df.gpu_id.unique():
        gpu_data = monitor_df[monitor_df.gpu_id == idx].iloc[0]
        device_info[f"device_{gpu_data.gpu_id}"] = str(gpu_data.gpu_name)
    
    return device_info


def parse_time_to_utc(start_time: str, duration_s: float)->tuple[str, str]:
    """
    Parse time to UTC.

    Args:
        start_time (str): Start time.
        duration_s (float): Duration in seconds.

    Returns:
        tuple[str, str]: Start and end time in UTC.
    """
    tzinfo = datetime.now().astimezone().tzinfo
    parsed_date = datetime.strptime(start_time, "%Y%m%d-%H%M%S")
    local_date = parsed_date.replace(tzinfo=tzinfo)
    start_utc_dt = local_date.astimezone(timezone.utc)
    end_utc_dt = start_utc_dt + timedelta(seconds=duration_s)
    return start_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_utc_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def filter_benchmark_report(report_json: dict, keys_to_keep: list[str])->dict:
    """
    Filter benchmark report.

    Args:
        report_json (dict): Benchmark report.
        keys_to_keep (list[str]): Keys to keep.

    Returns:
        dict: Filtered report.
    """
    filtered_report = {}
    for k in report_json.keys():
        if k in keys_to_keep:
            filtered_report[k] = report_json[k]
    return filtered_report

########################################################################################
# Benchmark Summary
########################################################################################

def upload_benchmark_summary(benchmark_summary_dir: str) -> list[CompleteBenchmarkRun]:
    """
    Format benchmark summary to upload to the database.

    Args:
        benchmark_summary_dir (str): Path to the benchmark summary directory.

    Returns:
        list[CompleteBenchmarkRun]: Benchmark summary list.   
    """
    benchmark_summary_files = Path(benchmark_summary_dir).glob("benchmark_display_id_*.md")
    benchmark_summary_list = []

    for benchmark_summary_file in benchmark_summary_files:
        summary_info_dict = parse_benchmark_summary_filename(benchmark_summary_file)
        df, description_dict = parse_benchmark_summary_table(benchmark_summary_file)
        records = df.to_dict(orient="records")
        for record in records:
            measurements = []
            for key, value in record.items():
                if key.strip() in ["ISL", "OSL", "Concurrency", "N Req"]:
                    continue
                measurement = BenchmarkMeasurement(
                    step_start_ts=summary_info_dict["date"],
                    step_end_ts=summary_info_dict["date"],
                    iteration=1,
                    step_name="benchmark_summary",
                    name=key.strip(),
                    value=float(value),
                    device_power=None,
                    device_temperature=None,
                )
                measurements.append(measurement)
                    
            benchmark_summary_list.append(CompleteBenchmarkRun(
                run_start_ts=summary_info_dict["date"],
                run_end_ts=summary_info_dict["date"],
                run_type="benchmark_summary",
                device_hostname=socket.gethostname(),
                ml_model_name=summary_info_dict["model_name"],
                measurements=measurements,
                training=False,
                input_sequence_length=record["ISL"],
                output_sequence_length=record["OSL"],
            ))
    return benchmark_summary_list

def parse_benchmark_summary_filename(markdown_report_file: str) -> dict:
    """
    Extract benchmark information from the filename.

    Args:
        markdown_report_file (str): Path to the markdown report file.

    Returns:
        dict: Benchmark summary info.
    """ 
    filename = Path(markdown_report_file).name
    pattern = r'benchmark_display_id_([^_]+)_([^_]+(?:-[^_]+)*)_([^_]+)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.md'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename doesn't match expected pattern: {filename}")

    framework, model_name, device, date_str = match.groups()
    date_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
    local_tz = datetime.now().astimezone().tzinfo
    date_obj = date_obj.replace(tzinfo=local_tz)
    utc_date_obj = date_obj.astimezone(timezone.utc)
        
    return {
            'framework': framework,
            'model_name': model_name,
            'device': device,
            'date': utc_date_obj,
        }

def parse_benchmark_summary_table(markdown_report_file: str) -> tuple[pd.DataFrame, dict]:
    """
    Parse benchmark markdown summary table.

    Args:
        markdown_report_file (str): Path to the markdown report file.

    Returns:
        tuple[pd.DataFrame, dict]: Benchmark summary table and description dictionary.
    """
    with open(markdown_report_file, "r") as f:
        markdown_report = f.readlines()
    
    table_lines = []
    description_dict = {}
    for line in markdown_report:
        stripped_line = line.strip()
        if stripped_line.startswith("|") and stripped_line.endswith("|"):
            table_lines.append(stripped_line)
        elif stripped_line.startswith(">"):
            clean_line = stripped_line.strip().lstrip('> ')
            key,value = clean_line.split(":")
            description_dict[key.strip()] = value.strip()
    if not table_lines:
        raise ValueError("No table lines found in markdown file")
    
    table_string = "\n".join(table_lines)
    df = pd.read_csv(
        StringIO(table_string),
        sep='|',
        header=0,
        skipinitialspace=True,
    ).dropna(axis=1, how='all')
    df.columns = df.columns.str.strip()
    if len(df) > 0:

        first_row_str = df.iloc[0].astype(str)
        if first_row_str.str.contains('---').any() or first_row_str.str.contains(':--').any():
            df = df.iloc[1:].reset_index(drop=True)

    return df, description_dict

########################################################################################
# Evals
########################################################################################

def convert_evals_time_to_utc(date:float,start_time: float, end_time: float) -> tuple[datetime, datetime]:
    """
    Convert evals time to UTC.

    Args:
        date (float): Benchmark date.
        start_time (float): Start time.
        end_time (float): End time.

    Returns:
        tuple[datetime, datetime]: Start and end time in UTC.
    """
    benchmark_date = date
    start_time = start_time
    end_time = end_time
    offset = benchmark_date - start_time
    absolute_start = start_time + offset
    absolute_end = end_time + offset
    start_dt = datetime.fromtimestamp(absolute_start, tz=timezone.utc)
    end_dt = datetime.fromtimestamp(absolute_end, tz=timezone.utc)
    return start_dt, end_dt

def parse_evals_results(evals_dict: dict) -> dict:
    """
    Filter evals results to keep only scores and stderr.

    Args:
        evals_dict (dict): Evals results dictionary.

    Returns:
        dict: Evals results dictionary.
    """
    results = evals_dict["results"].copy()
    evals_results = {}
    for k in results.keys():
        evals_results[k] = {}
        for key, value in results[k].items():
            if "alias" in key:
                continue
            elif "stderr" in key:
                evals_results[k]["stderr"] = value
            else:
                evals_results[k]["score"] = value
    return evals_results

def upload_evals_results(evals_results_dir: str) -> list[CompleteBenchmarkRun]:
    """
    Format evals results to upload to the database.

    Args:
        evals_results_dir (str): Path to the evals results directory.

    Returns:
        list[CompleteBenchmarkRun]: Evals results list.
    """
    evals_results_files = Path(evals_results_dir).glob("results_*.json")
    evals_results_list = []
    for evals_results_file in evals_results_files:
        with open(evals_results_file, "r") as f:
            evals_results = json.load(f)
        evals_scores = parse_evals_results(evals_results)
        start_time,end_time = convert_evals_time_to_utc(evals_results["date"],
                                                        evals_results["start_time"],
                                                        evals_results["end_time"])
        measurements = []
        for metric_name, metric_value in evals_scores.items():
            for key, value in metric_value.items():
                measurement = BenchmarkMeasurement(
                    step_start_ts=start_time,
                    step_end_ts=end_time,
                    iteration=1,
                    step_name=metric_name,
                    name=key,
                    value=float(value) if value != "N/A" else 0.0,
                    device_power=None,
                    device_temperature=None,
                )
                measurements.append(measurement)
        
        dataset_name = None
        for k in evals_results["configs"].keys():
            if "dataset_path" in evals_results["configs"][k] :
                dataset_name = evals_results["configs"][k]["dataset_path"]
                break

        evals_results_list.append(CompleteBenchmarkRun(
            run_start_ts=start_time,
            run_end_ts=end_time,
            run_type="evals",
            device_hostname=socket.gethostname(),
            ml_model_name=evals_results["model_name_sanitized"],
            measurements=measurements,
            training=False,
            dataset_name=dataset_name,
        ))
    return evals_results_list

