# Locust Load Testing

## Setup Environment

1. Create virtual environment

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2. Upgrade `pip` and install dependencies:

    ```bash
    pip install -U pip
    pip install -r requirements.txt
    ```

### Authorization

Export your `AUTHORIZATION` token as an environment variable. For instructions on generating your token, refer to the [JWT_TOKEN Authorization](../tt-metal-llama3-70b/README.md#jwt_token-authorization) guide for your deployed model.

```bash
export AUTHORIZATION="your_jwt_token_here"
```

## Test Types

### Static Test

Run a basic test with a single user and a predefined input prompt.

- Prompt: "What is in Austin Texas?"
  - Prompt Length: 7 tokens
  - Output Length: 128 tokens

Command to run the static test:

```bash
locust --config locust_static.conf
```

### Dynamic Test

Run a dynamic test using a dataset of variable input prompts.

- Dataset: [fka/awesome-chatgpt-prompts](https://huggingface.co/datasets/fka/awesome-chatgpt-prompts)

Command to run the dynamic test:

```bash
locust --config locust_dynamic.conf
```

## Test Configuration

You can configure Locust tests using `.conf` files. The key configurations to modify are:

- **`users`**: Maximum number of concurrent users.
- **`spawn-rate`**: Number of users spawned per second.
- **`run-time`**: Total duration of the test (e.g., `300s`, `20m`, `3h`, `1h30m`, etc.).

For more details, see the [Locust configuration guide](https://docs.locust.io/en/2.25.0/configuration.html).

### Example

To run a test with 32 users, all launched simultaneously, for a duration of 3 minutes, set the parameters:

```bash
users = 32
spawn-rate = 32
run-time = 3m
```
