from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

# Reference: https://github.com/tenstorrent/tt-github-actions/blob/main/.github/actions/collect_data/src/pydantic_models.py

class BenchmarkMeasurement(BaseModel):
    """
    Contains measurements for each benchmark run, iteration and step.

    A run can have multiple iterations, each iteration can have multiple steps and each
    step can execute multiple measurements.
    """

    step_start_ts: datetime = Field(description="Timestamp with time zone when the step started.")
    step_end_ts: datetime = Field(description="Timestamp with time zone when the step ended.")
    iteration: int = Field(
        description="A benchmark run can comprise a loop that repeats with the same "
        "parameters the same sequence of steps and measurements for each. "
        "This integer is the repetition number."
    )
    step_name: str = Field(description="Name of the benchmark step within the run.")
    step_warm_up_num_iterations: Optional[int] = Field(
        None, description="Number of iterations for device warm-up at each step."
    )
    name: str = Field(
        description="Name of the measurement performed, e.g. tokens_per_sec_per_user, "
        "tokens_per_sec, images_per_sec, pearson_correlation, "
        "top1/top5 ratios."
    )
    value: float = Field(description="Measured value.")
    target: Optional[float] = Field(None, description="Target value.")
    device_power: Optional[float] = Field(
        None,
        description="Average power consumption in Watts during the benchmark step.",
    )
    device_temperature: Optional[float] = Field(
        None, description="Average temperature of the device during the benchmark."
    )


class CompleteBenchmarkRun(BaseModel):
    """
    Contains information about each execution of an AI model benchmark, called benchmark
    run, composed of steps each of which performs a set of measurements.

    The sequence of steps in a run can be iterated in a loop.
    """

    run_start_ts: datetime = Field(description="Timestamp with time zone when the benchmark run started.")
    run_end_ts: datetime = Field(description="Timestamp with time zone when the benchmark run ended.")
    run_type: str = Field(description="Description of the benchmark run, e.g. a100_fp16_experiments.")
    git_repo_name: Optional[str] = Field(
        None,
        description="Name of the Git repository containing the code that executes " "the benchmark.",
    )
    git_commit_hash: Optional[str] = Field(
        None,
        description="Git commit hash of the code used to run the benchmark (software " "version info).",
    )
    git_commit_ts: Optional[datetime] = Field(None, description="Timestamp with timezone of the git commit.")
    git_branch_name: Optional[str] = Field(
        None, description="Name of the Git branch associated with the benchmark run."
    )
    github_pipeline_id: Optional[int] = Field(
        None,
        description="Unique identifier for the pipeline record from GitHub Actions.",
    )
    github_pipeline_link: Optional[str] = Field(
        None,
        description="Link to the GitHub job run associated with the benchmark run.",
    )
    github_job_id: Optional[int] = Field(None, description="Unique GitHub Actions CI job ID.")
    user_name: Optional[str] = Field(None, description="Name of the person that executed the benchmark run.")
    docker_image: Optional[str] = Field(
        None,
        description="Name or ID of the Docker image used for benchmarking (software "
        "version info), e.g., trt-llm-v080.",
    )
    device_hostname: str = Field(description="Host name of the device on which the benchmark is performed.")
    device_ip: Optional[str] = Field(None, description="Host IP address.")
    device_info: Optional[dict] = Field(
        None,
        description="Device information as JSON, such as manufacturer, card_type, "
        "dram_size, num_cores, price, bus_interface, optimal_clock_speed.",
    )
    ml_model_name: str = Field(description="Name of the benchmarked neural network model.")
    ml_model_type: Optional[str] = Field(
        None,
        description="Model type, such as text generation, classification, question " "answering, etc.",
    )
    num_layers: Optional[int] = Field(None, description="Number of layers of the model.")
    batch_size: Optional[int] = Field(None, description="Batch size.")
    config_params: Optional[dict] = Field(None, description="Additional training/inference parameters.")
    precision: Optional[str] = Field(
        None,
        description="Numerical precision, such as bfp8, fp16, or a mix such as " "fp16_act_bfp8_weights, etc.",
    )
    dataset_name: Optional[str] = Field(None, description="Name of the dataset used for the benchmark.")
    profiler_name: Optional[str] = Field(None, description="Profiler to time the benchmark.")
    input_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as input to the model, applicable " "to sequence models.",
    )
    output_sequence_length: Optional[int] = Field(
        None,
        description="Length of the sequence used as output by the model, applicable " "to sequence models.",
    )
    image_dimension: Optional[str] = Field(
        None,
        description="Dimension of the image, e.g. 224x224x3, applicable to computer " "vision models.",
    )
    perf_analysis: Optional[bool] = Field(
        None,
        description="If the model was run in perf analysis mode. This is " "kernel/operation execution mode.",
    )
    training: Optional[bool] = Field(None, description="ML model benchmarks for training or inference.")
    measurements: List[BenchmarkMeasurement] = Field(description="List of benchmark measurements.")
