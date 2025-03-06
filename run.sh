#!/bin/bash
# run.sh: A CLI for running workflows with optional docker, device, and workflow-args.
#
# Usage: ./run.sh {model} --workflow {benchmarks, evals, server} [--docker] [--device {N150, N300, T3K}] [--workflow-args "..."]

# Global configuration arrays for valid options.
valid_workflows=("benchmarks" "evals" "server")
valid_devices=("N150" "N300" "T3K")

# ------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------

# Display the usage information.
usage() {
    cat <<EOF
Usage: $0 {model}
  --workflow {${valid_workflows[*]}}
  [--docker]
  [--device {${valid_devices[*]}}]
  [--workflow-args 'param1=value1 param2=value2']
EOF
    exit 1
}

# Helper function to check if an element exists in an array.
contains_element() {
    local element
    for element in "${@:2}"; do
        if [[ "$element" == "$1" ]]; then
            return 0
        fi
    done
    return 1
}

# Parse command-line arguments and set global variables.
parse_arguments() {
    # The first positional parameter is the model.
    model="$1"
    shift

    # Initialize variables for the options.
    workflow=""
    docker_flag=0
    device=""
    workflow_args=""

    # Parse the remaining arguments.
    while [ $# -gt 0 ]; do
        case "$1" in
            --workflow)
                if [ -n "$2" ]; then
                    workflow="$2"
                    shift 2
                else
                    echo "Error: --workflow requires an argument."
                    exit 1
                fi
                ;;
            --docker)
                docker_flag=1
                shift
                ;;
            --device)
                if [ -n "$2" ]; then
                    device="$2"
                    shift 2
                else
                    echo "Error: --device requires an argument."
                    exit 1
                fi
                ;;
            --workflow-args)
                if [ -n "$2" ]; then
                    workflow_args="$2"
                    shift 2
                else
                    echo "Error: --workflow-args requires an argument."
                    exit 1
                fi
                ;;
            *)
                echo "Unknown option: $1"
                usage
                ;;
        esac
    done
}

# Validate parsed arguments using the configuration arrays.
validate_arguments() {
    # Ensure that the required --workflow option was provided.
    if [ -z "$workflow" ]; then
        echo "Error: --workflow option is required."
        usage
    fi

    # Validate the --workflow option using the valid_workflows config list.
    if ! contains_element "$workflow" "${valid_workflows[@]}"; then
        echo "Error: --workflow must be one of: ${valid_workflows[*]}"
        usage
    fi

    # Validate the --device option (if provided) using the valid_devices config list.
    if [ -n "$device" ]; then
        if ! contains_element "$device" "${valid_devices[@]}"; then
            echo "Error: --device must be one of: ${valid_devices[*]}"
            usage
        fi
    fi
}

# Execute the workflow based on the provided options.
run_workflow() {
    echo "Model:          $model"
    echo "Workflow:       $workflow"
    echo "Docker flag:    $docker_flag"
    echo "Device:         $device"
    echo "Workflow Args:  $workflow_args"
    echo "-------------------------------"

    case "$workflow" in
        benchmarks)
            echo "Running benchmarks..."
            # Insert your benchmark commands here.
            ;;
        evals)
            echo "Running evaluations..."
            # Insert your evaluation commands here.
            ;;
        server)
            echo "Starting server..."
            # Insert your server commands here.
            ;;
        *)
            echo "Error: Unknown workflow '$workflow'"
            exit 1
            ;;
    esac

    if [ "$docker_flag" -eq 1 ]; then
        echo "Docker mode enabled."
        # Insert docker related commands here.
    fi

    if [ -n "$workflow_args" ]; then
        echo "Additional workflow arguments: $workflow_args"
        # Process additional workflow arguments as needed.
    fi
}

# ------------------------------------------------------------------------------
# Main script logic
# ------------------------------------------------------------------------------

# Ensure script is being executed, not sourced.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "⛔ Error: This script is being sourced. Please execute it directly:"
    echo "chmod +x ./setup.sh && ./setup.sh"
    set +euo pipefail  # Unset 'set -euo pipefail' when sourcing.
    return 1  # 'return' works when sourced; 'exit' would terminate the shell.
fi

# Check if at least one argument is provided.
if [ $# -lt 1 ]; then
    usage
fi

# Parse command-line arguments.
parse_arguments "$@"

# Validate the parsed arguments.
validate_arguments

# Execute the workflow.
run_workflow

exit 0
