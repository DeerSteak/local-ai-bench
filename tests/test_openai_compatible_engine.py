import io
import json

import pytest

from scripts.runtime.engines.openai_compatible import OpenAICompatibleEngine, endpoint_config, selection_errors


def test_local_endpoint_normalization_and_validation():
    assert endpoint_config('http://127.0.0.1:18300/')["url"] == 'http://127.0.0.1:18300/v1'
    assert endpoint_config('http://[::1]:8000/v1/')["url"] == 'http://[::1]:8000/v1'
    for url in ('https://example.com/v1', 'http://user:secret@localhost', 'http://localhost/?key=secret', 'file:///tmp/api'):
        with pytest.raises(ValueError):
            endpoint_config(url)
    with pytest.raises(ValueError):
        endpoint_config('http://localhost', context=0)


@pytest.fixture
def engine():
    instance = OpenAICompatibleEngine(endpoint_config('http://localhost:18300'))
    instance._models = ['served-model']
    return instance


def test_discovery_auto_selects_one_and_preserves_exact_name(engine, monkeypatch):
    monkeypatch.setattr(engine, '_request', lambda *a: io.StringIO(json.dumps({'data': [{'id': 'owner/model'}]})))
    assert engine.list_installed_models()[0]['tag'] == 'owner/model'
    engine.settings['label'] = 'NVFP4 Hybrid'
    assert engine.list_installed_models()[0]['short'] == 'NVFP4 Hybrid'
    engine.settings['model'] = 'missing'
    with pytest.raises(ValueError, match='no longer'):
        engine.list_installed_models()


@pytest.mark.parametrize('payload', [{}, {'data': []}, {'data': [{'id': ''}]}])
def test_invalid_discovery_fails(engine, monkeypatch, payload):
    monkeypatch.setattr(engine, '_request', lambda *a: io.StringIO(json.dumps(payload)))
    with pytest.raises(ValueError):
        engine.discover_models()


def test_multiple_models_require_explicit_selection(engine, monkeypatch):
    monkeypatch.setattr(engine, '_request', lambda *a: io.StringIO(json.dumps({'data': [{'id': 'a'}, {'id': 'b'}]})))
    with pytest.raises(ValueError, match='multiple'):
        engine.list_installed_models()
    engine.settings['model'] = 'b'
    assert engine.list_installed_models()[0]['tag'] == 'b'


def test_lifecycle_never_requests_unload_or_process_control(engine, monkeypatch):
    def forbidden(*a, **k):
        pytest.fail('lifecycle must leave the server alone')
    monkeypatch.setattr(engine, '_request', forbidden)
    engine.stop()
    engine.unload('served-model')
    engine.unload_all()
    assert not engine.wait_until_unloaded('served-model')
    assert not engine.can_reset_model_state()
    assert engine.resume_artifact_paths('served-model') == ()
    assert engine.resume_runtime_paths() == {}


@pytest.mark.parametrize('usage', [None, {'completion_tokens': True, 'prompt_tokens': 5}, {'completion_tokens': -1, 'prompt_tokens': 5}])
def test_missing_or_invalid_usage_does_not_become_a_measurement(engine, monkeypatch, usage):
    chunks = [{'choices': [{'delta': {'content': 'hello'}}]}]
    if usage:
        chunks.append({'usage': usage})
    monkeypatch.setattr(engine, '_post', lambda *a: io.BytesIO(''.join('data: '+json.dumps(x)+'\n\n' for x in chunks).encode()))
    with pytest.raises(ValueError, match='usage'):
        engine.generate('served-model', 'hello')


def test_chat_stream_has_only_standard_fields_and_server_usage(engine, monkeypatch):
    captured = []
    def post(path, payload, timeout):
        captured.append(payload)
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"hello"},"finish_reason":"stop"}]}\n\ndata: {"usage":{"completion_tokens":2,"prompt_tokens":6}}\n\n')
    monkeypatch.setattr(engine, '_post', post)
    result = engine.generate('served-model', 'test')
    assert result.generated_tokens == 2 and result.prompt_tokens == 6
    assert result.response_text == 'hello' and result.model_load_sec == 0
    assert result.server_prompt_sec is None
    assert captured[0]['messages'] == [{'role': 'user', 'content': 'test'}]
    assert 'cache_salt' not in captured[0] and 'top_k' not in captured[0]


def test_workload_boundary():
    assert not selection_errors(['openai-compatible'], ['llm', 'conv', 'tool'])
    assert len(selection_errors(['openai-compatible', 'vllm'], ['vllmbench'], 'both', True)) == 4




def test_external_preflight_does_not_claim_clean_state(engine):
    from scripts.runtime.model_preflight import run_formatting_probe
    check = run_formatting_probe(engine, 'served-model', 1024, 30)
    assert check.severity == 'warning' and check.status == 'externally_managed'


def test_authentication_is_only_in_request_header(engine, monkeypatch):
    captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(request)
            return io.BytesIO(b'{}')
    monkeypatch.setenv('LOCAL_AI_BENCH_API_KEY', 'private-key')
    monkeypatch.setattr('urllib.request.build_opener', lambda *a: Opener())
    engine._request('/models')
    assert captured[0].get_header('Authorization') == 'Bearer private-key'
    assert 'private-key' not in json.dumps(engine.settings)




def test_interrupted_stream_preserves_partial_accuracy_response(engine, monkeypatch):
    from scripts.runtime.shared import EngineTimeout
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __iter__(self):
            yield b'data: {"choices":[{"delta":{"content":"Answer: B"}}]}\n'
            raise TimeoutError()
    monkeypatch.setattr(engine, '_post', lambda *args: Stream())
    with pytest.raises(EngineTimeout) as caught:
        engine.chat('served-model', [{'role': 'user', 'content': 'Choose'}])
    assert caught.value.partial_text == 'Answer: B'


def test_standard_tool_fragments_are_assembled(engine, monkeypatch):
    rows = [
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'name': 'weather', 'arguments': '{"city":'}}]}}]},
        {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"Paris"}'}}]}, 'finish_reason': 'tool_calls'}]},
        {'usage': {'completion_tokens': 8, 'prompt_tokens': 12}},
    ]
    monkeypatch.setattr(engine, '_post', lambda *a: io.BytesIO(''.join('data: '+json.dumps(row)+'\n' for row in rows).encode()))
    result = engine.chat_tools('served-model', [{'role': 'user', 'content': 'Weather?'}], [])
    assert result.tool_calls == [{'name': 'weather', 'arguments': {'city': 'Paris'}}]
