from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "serve_realtime_ws.py"


def _source_text():
    return SOURCE.read_text(encoding="utf-8")


def test_realtime_server_exposes_long_session_controls():
    source = _source_text()

    assert 'parser.add_argument("--disable-spk"' in source
    assert 'parser.add_argument("--ws-ping-interval"' in source
    assert 'parser.add_argument("--ws-ping-timeout"' in source
    assert "ping_interval=args.ws_ping_interval" in source
    assert "ping_timeout=args.ws_ping_timeout" in source


def test_realtime_server_keeps_heavy_session_work_off_event_loop():
    source = _source_text()

    assert "None if args.disable_spk else HybridSpeakerTracker" in source
    assert "await asyncio.to_thread(session.add_audio, message)" in source
    assert "await asyncio.to_thread(session.decode, is_final=True)" in source
    assert "await asyncio.to_thread(session.decode, is_final=False)" in source
