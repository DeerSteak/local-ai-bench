import json

import pytest

from scripts.runtime.engines.openai_compatible import API_ENV, OpenAICompatibleEngine, endpoint_config


@pytest.fixture
def engine():
    instance = OpenAICompatibleEngine(endpoint_config("http://localhost:18300"))
    instance._models = ["served-model"]
    return instance

def test_discovered_model_reaches_plan_and_worker_without_local_catalog(engine, monkeypatch):
    from scripts.app.benchmark import resolve_engine_scopes, selected_plan_models
    from scripts.results.run_plan import RunPlan
    from scripts.runtime.workload_runner import configure_runner_engine
    from scripts.workloads.methodology_profile import resolve_methodology_profile
    monkeypatch.setattr(engine, 'discover_models', lambda: ['served-model'])
    engine.settings['label'] = 'RadixArk NVFP4 Hybrid'
    scopes, errors = resolve_engine_scopes(['openai-compatible'], lambda _: engine, [], 'all', None, ['llm', 'conc_chat'])
    assert not errors
    models = scopes[0]['llm_models']
    assert models == scopes[0]['concurrency_models']
    assert models[0]['short'] == 'RadixArk NVFP4 Hybrid'
    profile = resolve_methodology_profile(engine_name=engine.name, tests=['llm'], cpu_only=False)
    settings = {'warmup_runs': 0, 'cpu_only': False, 'force_all': False,
                'external_api': engine.settings, 'methodology_profile': profile['profile'],
                'effective_optimizations': profile['effective_optimizations'], 'sampling_profile': profile['sampling_profile']}
    plan = RunPlan.create(application_version='6.1.1', engine_name=engine.name, tests=['llm'], stage_order=['llm'],
                          models=selected_plan_models(['llm'], models, [], [], []), effective_config=settings)
    assert plan.execution_identity['runtime']['external_api'] == engine.settings
    restored = OpenAICompatibleEngine()
    configure_runner_engine(restored, 'cuda', False, plan.effective_config)
    assert restored.settings == engine.settings
    assert 'externally_managed' in json.dumps(plan.to_dict())
    with pytest.raises(ValueError, match='credentials'):
        RunPlan.create(application_version='6.1.1', engine_name=engine.name, tests=['llm'], stage_order=['llm'],
                       models=plan.models, effective_config={**settings, 'external_api': {**engine.settings, 'api_key': 'secret'}})


def test_endpoint_options_reach_cli_without_local_model_selection():
    from scripts.app.benchmark_frontend import build_benchmark_command, GUI_OPTION_DEFAULTS
    from pathlib import Path
    options = {**GUI_OPTION_DEFAULTS, 'api_base_url': 'http://localhost:18300/v1', 'api_label': 'Hybrid'}
    command = build_benchmark_command('openai-compatible', Path('/unused'), ['llm'], [], gui_options=options)
    assert command[command.index('--api-base-url') + 1] == options['api_base_url']
    assert '--llm-models' not in command and '--comfyui' not in command
    assert 'LOCAL_AI_BENCH_API_KEY' not in ' '.join(command)


def test_gui_can_launch_without_any_local_models():
    from scripts.app.benchmark_gui import prepare_benchmark_launch, BenchmarkLaunchReady
    from scripts.app.benchmark_frontend import GUI_OPTION_DEFAULTS
    from pathlib import Path
    result = prepare_benchmark_launch(
        engine='openai-compatible', tests=['llm', 'conv'], entries=[], model_owners={},
        max_prompt_tokens=8192, tg_tokens=[], gui_options={**GUI_OPTION_DEFAULTS, 'api_base_url': 'http://localhost:18300'},
        selected_preset='Custom', detected_tools={}, found_comfyui=None, detected_comfyui=Path('/unused'),
    )
    assert isinstance(result, BenchmarkLaunchReady), result


def test_external_telemetry_does_not_mislabel_client_rss(monkeypatch):
    from scripts.runtime.telemetry import CaseTelemetry
    monkeypatch.setattr('scripts.runtime.telemetry.process_resource_usage', lambda pid: pytest.fail('client RSS must not be sampled'))
    monkeypatch.setattr('scripts.runtime.telemetry.system_memory_usage', lambda: (60, 120))
    telemetry = CaseTelemetry(include_process_memory=False, sources={'host_ram_used_gb': 'psutil'})
    reading = telemetry.sampler._sample(0, 'measured')
    assert reading.host_ram_used_gb == 60 and reading.process_rss_gb is None
    assert telemetry.sources['process_rss_gb'] == 'unavailable'


def test_external_plan_cannot_resume_unverified_server_identity(tmp_path):
    from scripts.results.run_plan import RunPlan
    from scripts.results.event_store import EventStore
    plan = RunPlan.create(application_version='6.1.1', engine_name='openai-compatible', tests=['llm'], stage_order=['llm'],
                          models={'llm': [{'tag': 'served', 'short': 'served'}]},
                          effective_config={'warmup_runs': 0, 'cpu_only': False, 'force_all': False,
                                            'external_api': endpoint_config('http://localhost:18300')})
    store = EventStore(tmp_path / 'events.sqlite')
    try:
        store.create_job(plan)
        with pytest.raises(ValueError, match='unverified'):
            store.resume_job(plan.job_id)
        with pytest.raises(ValueError, match='unverified'):
            store.prepare_recovery(plan.job_id, 'llm')
    finally:
        store.close()


def test_terminal_endpoint_run_discovers_model_and_keeps_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv(API_ENV, "{}")
    from scripts.app.benchmark_frontend import run_frontend, load_frontend_state
    choices = iter(['http://localhost:18300/v1', 'Hybrid', '32768', '', '', ''])
    captured = []
    class Endpoint:
        def list_installed_models(self):
            return [{'tag': 'served-model', 'label': 'Hybrid', 'short': 'Hybrid'}]
    from unittest.mock import patch
    with patch.object(OpenAICompatibleEngine, 'discover_models', return_value=['served-model']):
        result = run_frontend(
            input_fn=lambda _=None: next(choices), output_fn=lambda _: None, clear_fn=lambda: None,
            process_runner=lambda command: captured.append(command) or 0,
            engine_names_fn=lambda: ['openai-compatible'], engine_factory=lambda _: Endpoint(),
            inventory_builder=lambda engine, path: {'llm': [], 'custom': engine.list_installed_models(), 'embedding': [], 'image': []},
            state_path=tmp_path / 'state.json',
        )
    assert result == 0
    assert '--api-base-url' in captured[0]
    state = load_frontend_state(tmp_path / 'state.json')
    assert state is not None
    assert state['gui_options']['api_label'] == 'Hybrid'
