import subprocess
import sys
import os
from pathlib import Path

from workflows.logger import get_logger
from workflows.configs import (
    workflow_config_map,
    get_repo_root_path,
    WorkflowType,
    get_default_workflow_root_log_dir,
)

logger = get_logger()


def run_command(command, shell=True, check=True):
    logger.info("Running command: %s", command)
    result = subprocess.run(
        command, shell=True, check=check, text=True, capture_output=True
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.error(result.stderr)
    return result


class WorkflowSetup:
    def __init__(self, args):
        self.args = args
        self.workflow_type = WorkflowType.from_string(args.workflow)
        self.workflow_config = workflow_config_map[self.workflow_type]
        self.workflow_dir = get_repo_root_path() / "workflows"

    def boostrap_uv(self):
        # Step 1: Check Python version
        python_version = sys.version_info
        if python_version < (3, 6):
            logger.error("Python 3.6 or higher is required.")
            sys.exit(1)
        logger.info(
            "Python version: %d.%d.%d",
            python_version.major,
            python_version.minor,
            python_version.micro,
        )

        # Step 2: Create a virtual environment
        venv_name = ".venv_setup_workflow"
        venv_dir = self.workflow_dir / venv_name
        uv_exec = venv_dir / "bin" / "uv"
        if not venv_dir.exists():
            logger.info("Creating virtual environment in '%s'...", venv_dir)
            run_command(f"{sys.executable} -m venv {venv_dir}")
            # Step 3: Install 'uv' using pip
            # Note: Activating the virtual environment in a script doesn't affect the current shell,
            # so we directly use the pip executable from the venv.
            pip_exec = venv_dir / "bin" / "pip"

            logger.info("Installing 'uv' using pip...")
            run_command(f"{pip_exec} install uv")

            logger.info("uv bootsrap installation complete.")
            # check version
            run_command(f"{str(uv_exec)} --version")

        self.uv_exec = uv_exec

    def create_workflow_venv(self):
        workflow_dir = self.workflow_config["run_script_path"].parent
        venv_name = f".venv_{self.workflow_config['name']}"
        self.workflow_venv_path = workflow_dir / venv_name
        self.workflow_venv_pip_exec = self.workflow_venv_path / "bin" / "pip"
        self.workflow_venv_python_exec = self.workflow_venv_path / "bin" / "python"
        if not self.workflow_venv_path.exists():
            python_version = self.workflow_config["python_version"]
            run_command(
                f"{str(self.uv_exec)} venv --python={python_version} {self.workflow_venv_path}"
            )
            run_command(f"{self.workflow_venv_python_exec} -m ensurepip --default-pip")
            run_command(f"{self.workflow_venv_pip_exec} pip install --upgrade pip")

    def setup_llama_cookbook(self):
        cookbook_dir = self.workflow_venv_path / "llama-cookbook"
        if cookbook_dir.is_dir():
            logger.info(f"The directory {cookbook_dir} exists.")
        else:
            logger.info("The directory ~/llama-recipes does not exist. Setting up ...")
            original_dir = os.getcwd()
            # Clone the repository
            clone_cmd = f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
            run_command(clone_cmd)
            # Upgrade pip and setuptools
            run_command(f"{self.workflow_venv_pip_exec} install -U pip setuptools")
            # Install the package in editable mode
            os.chdir(cookbook_dir)
            run_command(f"{self.workflow_venv_pip_exec} install -e .")
            # Install specific dependencies
            run_command(
                f"{self.workflow_venv_pip_exec} install -U antlr4_python3_runtime==4.11"
            )
            logger.warning(
                "this might take 5 to 15+ minutes to install on first run ..."
            )
            run_command(
                f"{self.workflow_venv_pip_exec} install lm-eval[math,ifeval,sentencepiece,vllm]==0.4.3 pyjwt==2.7.0 pillow==11.1"
            )
            # Change directory to meta_eval and run the preparation script
            self.meta_eval_dir = (
                cookbook_dir
                / "end-to-end-use-cases"
                / "benchmarks"
                / "llm_eval_harness"
                / "meta_eval"
            )
            prepare_meta_eval = self.meta_eval_dir / "prepare_meta_eval.py"
            config_path = self.meta_eval_dir / "eval_config.yaml"
            os.chdir(self.meta_eval_dir)
            run_command(
                f"{self.workflow_venv_python_exec} {prepare_meta_eval} --config_path {config_path}"
            )
            os.chdir(original_dir)

    def setup_lm_eval(self):
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        run_command(
            f"{self.workflow_venv_pip_exec} install lm-eval==0.4.8 pyjwt==2.7.0 pillow==11.1"
        )

    def setup_evals(self):
        # TODO: map to model and eval tasks
        use_meta = False
        if use_meta:
            self.setup_llama_cookbook()
        else:
            self.setup_lm_eval()

    def setup_benchmarks(self):
        pass

    def setup_tests(self):
        pass

    def setup_workflow(self):
        self.create_workflow_venv()
        if self.workflow_type == WorkflowType.BENCHMARKS:
            self.setup_benchmarks()
        elif self.workflow_type == WorkflowType.EVALS:
            self.setup_evals()
        elif self.workflow_type == WorkflowType.TESTS:
            self.setup_tests()

    def get_jwt_secret(self):
        """
        Returns the JWT secret from the JWT_SECRET environment variable,
        or if not set, from the args.jwt_secret attribute.
        """
        jwt_secret = os.getenv("JWT_SECRET")
        if jwt_secret:
            return jwt_secret
        return getattr(self.args, "jwt_secret", None)

    def get_server_port(self):
        """
        Returns the server port from the SERVICE_PORT environment variable,
        or if not set, from the args.server_port attribute.
        """
        server_port = os.getenv("SERVICE_PORT")
        if server_port:
            return server_port
        return getattr(self.args, "server_port", 7000)

    def get_output_paths(self):
        root_log_dir = get_default_workflow_root_log_dir()
        output_path = root_log_dir / "eval_output"
        log_path = root_log_dir / "run_evals_logs"
        return output_path, log_path

    def run_script(self):
        script_path = self.workflow_config["run_script_path"]
        model = self.workflow_config["HF_MODEL_REPO_ID"]
        model_arg = f"--model {model}"
        output_path, log_path = self.get_output_paths()
        output_path_arg = f"--output_path {output_path}"
        log_path_arg = f"--log_path {log_path}"
        # optional args
        jwt_arg = f"--jwt-secret {self.get_jwt_secret()}" if self.get_jwt_secret else ""
        server_port_arg = (
            f"--server-port {self.get_server_port()}" if self.get_server_port() else ""
        )

        cmd = (
            f"{self.workflow_venv_python_exec} {script_path} "
            f"{model_arg} {output_path_arg} {log_path_arg} {jwt_arg} {server_port_arg}"
        )
        run_command(cmd)


def run_local(args):
    manager = WorkflowSetup(args)
    manager.boostrap_uv()
    manager.setup_workflow()
    manager.run_script()
