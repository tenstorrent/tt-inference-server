import json
import argparse
from datetime import datetime

def analyze_report(report_data):
    """Analyzes test results to create a summary."""
    summary = {}
    
    # Check if parameter_support exists and is a dictionary
    if "parameter_support" not in report_data or not isinstance(report_data["parameter_support"], dict):
        print("Warning: 'parameter_support' key missing or invalid in report.json.")
        return summary
    for test_case, tests in report_data["parameter_support"].items():
        if not tests:
            summary[test_case] = {
                "status": "SKIP",
                "details": "No tests run."
            }
            continue

        total_tests = len(tests)
        passed_tests = sum(1 for t in tests if t["status"] == "pass")
        failed_tests = total_tests - passed_tests

        if failed_tests > 0:
            status = "FAIL"
            # Find the first failure message
            first_failure = next((t for t in tests if t["status"] == "fail"), None)
            details = f"{passed_tests}/{total_tests} passed. First failure ({first_failure['test_node_name']}): {first_failure['message']}"
            # Truncate long error messages
            if len(details) > 200:
                details = details[:200] + "..."
        else:
            status = "PASS"
            details = f"{passed_tests}/{total_tests} tests passed."
            
        summary[test_case] = {
            "status": status,
            "details": details
        }
    return summary

def format_metadata(report_data):
    """Creates a Markdown table for the report metadata."""
    lines = [
        "### Test Run Metadata",
        "",
        "| Attribute | Value |",
        "| --- | --- |",
        f"| **Model Name** | `{report_data.get('model_name', 'N/A')}` |",
        f"| **Model Backend** | `{report_data.get('model_backend', 'N/A')}` |",
        f"| **Endpoint URL** | `{report_data.get('endpoint_url', 'N/A')}` |"
    ]
    
    timestamp = report_data.get('test_run_timestamp_utc', 'N/A')
    if timestamp != 'N/A':
        try:
            # Parse ISO timestamp and format it
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            pass # Keep original string if parsing fails
            
    lines.append(f"| **Test Timestamp** | {timestamp} |")
    lines.append("")
    return "\n".join(lines)

def format_summary_table(summary):
    """Creates the main results table."""
    lines = [
        "### Parameter Conformance Summary",
        "",
        "| Test Case | Status | Details |",
        "| --- | :---: | --- |"
    ]
    
    # Sort test cases alphabetically for consistent reports
    for test_case in sorted(summary.keys()):
        result = summary[test_case]
        status_emoji = "✅" if result['status'] == 'PASS' else ("❌" if result['status'] == 'FAIL' else "⚠️")
        lines.append(f"| `{test_case}` | {status_emoji} {result['status']} | {result['details']} |")
        
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Convert API test report JSON to Markdown.")
    parser.add_argument(
        "report_file", 
        nargs="?",
        default="report.json",
        help="Path to the input report.json file (default: report.json)"
    )
    parser.add_argument(
        "-o", "--output",
        default="report.md",
        help="Path to the output report.md file (default: report.md)"
    )
    args = parser.parse_args()

    try:
        with open(args.report_file, 'r') as f:
            report_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.report_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.report_file}")
        return

    # Analyze and format
    summary = analyze_report(report_data)
    metadata_md = format_metadata(report_data)
    summary_md = format_summary_table(summary)
    
    # Write to output file
    with open(args.output, 'w') as f:
        f.write("# LLM API Conformance Report\n\n")
        f.write(metadata_md)
        f.write(summary_md)

    print(f"Successfully generated Markdown report at {args.output}")

if __name__ == "__main__":
    main()
