# SPDX-License-Identifier: Apache-2.0
"""Exercise the real Harbor console entrypoint outside the checkout."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from llm_module.agentic import harbor

CONSOLE = Path(sys.executable).with_name('harbor')
REPO = Path(harbor.__file__).resolve().parents[2]


def config(tmp_path, agent, import_path):
    return harbor.HarborRunConfig(task_name='host_import_probe',dataset='unused',
        agent=agent,model_name='m',jobs_dir=tmp_path/'jobs',api_base='http://127.0.0.1:8000/v1',
        n_concurrent_trials=1,n_attempts=1,environment_type='docker',agent_kwargs={},n_tasks=2,
        override_cpus=None,override_memory_mb=None,timeout_multiplier=None,agent_timeout_sec=21600,
        venv_python=Path(sys.executable),agent_import_path=import_path)


@pytest.mark.skipif(not CONSOLE.exists(),reason='Pinned Harbor console required in test interpreter')
@pytest.mark.parametrize('agent, import_path',[
    ('terminus-2','llm_module.agentic.qwen38_isolated_agent:IsolatedTerminus2'),
    ('mini-swe-agent','llm_module.agentic.qwen38_isolated_agent:IsolatedMiniSweAgent')])
def test_real_console_custom_agent_uses_child_repo_path(tmp_path,monkeypatch,agent,import_path):
    unrelated=tmp_path/'unrelated';unrelated.mkdir()
    inherited=tmp_path/'inherited';inherited.mkdir()
    monkeypatch.setenv('PYTHONPATH',str(inherited))
    monkeypatch.setenv('TT_CHILD_TEST_PRESERVED','yes')
    clean_env=os.environ.copy()
    command=[str(CONSOLE),'agents','schema',import_path,'--json']
    # A pytest import does not make the independently launched console importable.
    before=subprocess.run(command,cwd=unrelated,env=clean_env,capture_output=True,text=True,timeout=30)
    assert before.returncode!=0
    assert "No module named 'llm_module'" in " ".join((before.stdout + before.stderr).split())
    captured={}
    def console_probe(cmd,**kwargs):
        assert cmd[0]==str(CONSOLE) and cmd[1]=='run'
        env=kwargs['env'];captured.update(env)
        assert env['PYTHONPATH']==str(REPO)+os.pathsep+str(inherited)
        assert env['TT_CHILD_TEST_PRESERVED']=='yes'
        assert env['OPENAI_BASE_URL']=='http://127.0.0.1:8000/v1'
        assert kwargs['cwd'] is None
        result=subprocess.run(command,cwd=unrelated,env=env,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stdout+result.stderr
        native=subprocess.run([str(CONSOLE),'agents','schema',agent,'--json'],cwd=unrelated,
            env=clean_env,capture_output=True,text=True,timeout=30)
        assert native.returncode==0,native.stdout+native.stderr
        assert json.loads(result.stdout)==json.loads(native.stdout)
        return 17
    monkeypatch.setattr(harbor,'run_with_progress',console_probe)
    assert harbor.run(config(tmp_path,agent,import_path))==17
    assert os.environ['PYTHONPATH']==str(inherited)  # Parent environment untouched.


def test_builtin_agent_child_path_is_unchanged(tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/inherited/a'+os.pathsep+'/inherited/b')
    def no_launch(cmd,**kwargs):
        assert kwargs['env']['PYTHONPATH']==os.environ['PYTHONPATH']
        return 17
    monkeypatch.setattr(harbor,'run_with_progress',no_launch)
    assert harbor.run(config(tmp_path,'terminus-2',None))==17


@pytest.mark.parametrize("inherited", [None, "", "/inherited/a:/inherited/b"])
def test_isolated_child_path_preserves_inherited_components(tmp_path, monkeypatch, inherited):
    if inherited is None:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    else:
        monkeypatch.setenv("PYTHONPATH", inherited)
    def no_launch(cmd, **kwargs):
        expected = str(REPO) if inherited is None else str(REPO) + os.pathsep + inherited
        assert kwargs["env"]["PYTHONPATH"] == expected
        return 17
    monkeypatch.setattr(harbor, "run_with_progress", no_launch)
    assert harbor.run(config(tmp_path, "terminus-2", harbor._ISOLATED_AGENTS[1])) == 17
    assert os.environ.get("PYTHONPATH") == inherited
