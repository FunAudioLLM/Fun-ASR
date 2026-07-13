import asyncio
import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import torch


SERVICE_PATH = Path(__file__).resolve().parents[1] / "serve_realtime_ws.py"


def load_service_module(monkeypatch):
    for package_name in (
        "funasr",
        "funasr.models",
        "funasr.models.fsmn_vad_streaming",
    ):
        package = types.ModuleType(package_name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, package_name, package)

    dynamic_vad_stub = types.ModuleType("funasr.models.fsmn_vad_streaming.dynamic_vad")
    dynamic_vad_stub.DynamicStreamingVAD = object
    monkeypatch.setitem(
        sys.modules,
        "funasr.models.fsmn_vad_streaming.dynamic_vad",
        dynamic_vad_stub,
    )
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(
            exceptions=types.SimpleNamespace(ConnectionClosed=Exception),
            serve=lambda *args, **kwargs: None,
        ),
    )

    module_name = "serve_realtime_ws_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVICE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SilentVad:
    current_speech_start = None

    def feed(self, audio, is_final=False):
        return []

    def reset(self):
        pass


class SegmentVad(SilentVad):
    def __init__(self, sample_rate, segment_end_sample):
        self.sample_rate = sample_rate
        self.segment_end_sample = segment_end_sample
        self.total_samples = 0

    def feed(self, audio, is_final=False):
        self.total_samples += len(audio)
        if self.total_samples == self.segment_end_sample:
            end_ms = int(self.total_samples * 1000 / self.sample_rate)
            return [[end_ms - 1000, end_ms]]
        return []


class DummyTokenizer:
    def encode(self, text):
        return []

    def decode(self, token_ids, skip_special_tokens=True):
        return ""


class DummyEngine:
    def __init__(self):
        self._engine = types.SimpleNamespace(tokenizer=DummyTokenizer())
        self.input_lengths = []

    def generate(self, inputs, **kwargs):
        self.input_lengths.extend(len(audio) for audio in inputs)
        return [{"text": "hello"}]


def test_two_hour_session_keeps_audio_bounded_and_duration_absolute(monkeypatch):
    module = load_service_module(monkeypatch)
    sample_rate = 10
    session = module.RealtimeASRSession(
        vllm_engine=DummyEngine(),
        asr_kwargs={},
        vad=SilentVad(),
        sample_rate=sample_rate,
        chunk_ms=1000,
        audio_lookback_sec=5,
    )
    one_second = np.zeros(sample_rate, dtype=np.int16).tobytes()

    for _ in range(2 * 60 * 60):
        session.add_audio(one_second)

    assert session.total_samples == 2 * 60 * 60 * sample_rate
    assert len(session.audio_buffer) <= 5 * sample_rate
    assert session.audio_buffer_start_sample == session.total_samples - len(session.audio_buffer)
    assert session._build_response(is_final=False)["duration_ms"] == 2 * 60 * 60 * 1000


def test_completed_segment_uses_absolute_offsets_after_audio_compaction(monkeypatch):
    module = load_service_module(monkeypatch)
    sample_rate = 16000
    vad = SegmentVad(sample_rate=sample_rate, segment_end_sample=11 * sample_rate)
    engine = DummyEngine()
    session = module.RealtimeASRSession(
        vllm_engine=engine,
        asr_kwargs={},
        vad=vad,
        sample_rate=sample_rate,
        chunk_ms=1000,
        audio_lookback_sec=2,
    )
    one_second = np.zeros(sample_rate, dtype=np.int16).tobytes()

    for _ in range(11):
        session.add_audio(one_second)

    assert engine.input_lengths == [sample_rate]
    assert session.locked_sentences == [{"text": "hello", "start": 10000, "end": 11000}]
    assert session._build_response(is_final=False)["duration_ms"] == 11000
    assert session.audio_buffer_start_sample > 0


def test_partial_decode_uses_recent_window(monkeypatch):
    module = load_service_module(monkeypatch)

    class ActiveVad(SilentVad):
        current_speech_start = 0

    sample_rate = 16000
    session = module.RealtimeASRSession(
        vllm_engine=DummyEngine(),
        asr_kwargs={},
        vad=ActiveVad(),
        sample_rate=sample_rate,
        partial_window_sec=2,
    )
    session.audio_buffer = np.arange(sample_rate * 5, dtype=np.float32)
    session.total_samples = len(session.audio_buffer)

    audio, start_ms = session.get_partial_decode_audio()

    assert start_ms == 3000
    np.testing.assert_array_equal(audio, session.audio_buffer[-sample_rate * 2:])


def test_final_decode_releases_audio_without_resetting_duration(monkeypatch):
    module = load_service_module(monkeypatch)
    sample_rate = 10
    for sample_count in (5, 100):
        session = module.RealtimeASRSession(
            vllm_engine=DummyEngine(),
            asr_kwargs={},
            vad=SilentVad(),
            sample_rate=sample_rate,
            chunk_ms=1000,
            audio_lookback_sec=5,
        )
        session.add_audio(np.zeros(sample_count, dtype=np.int16).tobytes())

        response = session.decode(is_final=True)

        assert response["duration_ms"] == sample_count * 100
        assert len(session.audio_buffer) == 0
        assert session.audio_buffer_start_sample == session.total_samples


def test_speaker_history_and_identity_state_have_hard_limits(monkeypatch):
    utils_stub = types.ModuleType("funasr.models.campplus.utils")
    utils_stub.sv_chunk = lambda segments: segments

    def postprocess(segments, vad_segments, labels, embeddings, return_spk_center=False):
        output = [[segment[0], segment[1], int(label)] for segment, label in zip(segments, labels)]
        centers = torch.stack(
            [embeddings[labels == label].mean(0) for label in sorted(set(labels.tolist()))]
        )
        return (output, centers) if return_spk_center else output

    def distribute_spk(sentences, speaker_segments):
        for sentence in sentences:
            sentence["spk"] = int(speaker_segments[-1][2])
        return sentences

    utils_stub.postprocess = postprocess
    utils_stub.distribute_spk = distribute_spk
    cluster_stub = types.ModuleType("funasr.models.campplus.cluster_backend")

    class FakeClusterBackend:
        def __init__(self, merge_thr):
            pass

        def to(self, device):
            return self

        def __call__(self, embeddings, oracle_num=None):
            return np.zeros(len(embeddings), dtype=np.int64)

    cluster_stub.ClusterBackend = FakeClusterBackend
    monkeypatch.setitem(sys.modules, "funasr.models.campplus.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "funasr.models.campplus.cluster_backend", cluster_stub)
    module = load_service_module(monkeypatch)

    class FakeSpeakerModel:
        def generate(self, input, **kwargs):
            return [{"spk_embedding": torch.tensor([[1.0, 0.0]])} for _ in input]

    tracker = module.HybridSpeakerTracker(
        spk_model=FakeSpeakerModel(),
        device="cpu",
        max_history_chunks=3,
        max_speakers=2,
    )
    tracker.sv_chunk = lambda segments: segments
    tracker.cluster_backend = lambda embeddings, oracle_num=None: np.zeros(
        len(embeddings), dtype=np.int64
    )

    for index in range(6):
        sentence = {"text": f"segment {index}", "start": index * 1000, "end": (index + 1) * 1000}
        tracker.assign_streaming(
            np.ones(16000, dtype=np.float32),
            index,
            index + 1,
            sentence,
        )

    tracker._map_cluster_centers(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), update=True)
    tracker._map_cluster_centers(torch.tensor([[-1.0, 0.0]]), update=True)

    assert len(tracker.all_chunks) == 3
    assert len(tracker.all_embeddings) == 3
    assert all(len(chunk) == 2 for chunk in tracker.all_chunks)
    assert len(tracker.speaker_centers) == 2

    finalized = tracker.finalize(
        [{"text": "abcdef", "start": 0, "end": 6000, "spk": 1}]
    )
    assert finalized == [
        {"text": "abc", "start": 0, "end": 3000, "spk": 1},
        {"text": "def", "start": 3000, "end": 6000, "spk": 0},
    ]


def test_handler_is_responsive_and_serializes_shared_model_work(monkeypatch):
    module = load_service_module(monkeypatch)

    class BlockingSession:
        active_workers = 0
        max_active_workers = 0
        worker_lock = threading.Lock()

        def __init__(self, *args, **kwargs):
            self.is_active = False

        def reset(self):
            pass

        def add_audio(self, message):
            with self.worker_lock:
                type(self).active_workers += 1
                type(self).max_active_workers = max(
                    type(self).max_active_workers,
                    type(self).active_workers,
                )
            try:
                time.sleep(0.2)
            finally:
                with self.worker_lock:
                    type(self).active_workers -= 1

        def should_decode(self):
            return False

    class FakeWebSocket:
        remote_address = ("127.0.0.1", 12345)

        def __init__(self):
            self.messages = iter(["START", b"audio"])
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration as error:
                raise StopAsyncIteration from error

        async def send(self, message):
            self.sent.append(message)

    monkeypatch.setattr(module, "load_models", lambda args: (object(), {}, object(), None))
    monkeypatch.setattr(module, "DynamicStreamingVAD", lambda model: object())
    monkeypatch.setattr(module, "RealtimeASRSession", BlockingSession)

    async def exercise_handler():
        stop = False
        gaps = []

        async def ticker():
            previous = asyncio.get_running_loop().time()
            while not stop:
                await asyncio.sleep(0.005)
                current = asyncio.get_running_loop().time()
                gaps.append(current - previous)
                previous = current

        ticker_task = asyncio.create_task(ticker())
        await asyncio.sleep(0.01)
        args = types.SimpleNamespace(
            device="cpu",
            decode_interval=0.48,
            disable_spk=True,
            partial_window_sec=15.0,
        )
        await asyncio.gather(
            module.handle_client(FakeWebSocket(), args),
            module.handle_client(FakeWebSocket(), args),
        )
        stop = True
        await ticker_task
        return gaps

    gaps = asyncio.run(exercise_handler())

    assert gaps
    assert max(gaps) < 0.08
    assert BlockingSession.max_active_workers == 1
