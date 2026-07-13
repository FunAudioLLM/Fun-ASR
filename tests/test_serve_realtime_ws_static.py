from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "serve_realtime_ws.py"


def _source_text():
    return SOURCE.read_text(encoding="utf-8")


def test_realtime_server_exposes_long_session_controls():
    source = _source_text()

    assert 'parser.add_argument("--disable-spk"' in source
    assert '"--partial-window-sec"' in source
    assert 'parser.add_argument("--ws-ping-interval"' in source
    assert 'parser.add_argument("--ws-ping-timeout"' in source
    assert 'parser.add_argument("--ws-close-timeout"' in source
    assert 'parser.add_argument("--ws-max-size"' in source
    assert "build_websocket_serve_kwargs(args)" in source
    assert "ping_interval" in source
    assert "ping_timeout" in source
    assert "close_timeout" in source
    assert "max_size" in source


def test_realtime_server_allows_disabling_websocket_pings():
    source = _source_text()

    assert "def _positive_or_none" in source
    assert "return None if value <= 0 else value" in source
    assert "_positive_or_none(args.ws_ping_interval)" in source
    assert "_positive_or_none(args.ws_ping_timeout)" in source
    assert "<=0 disables" in source


def test_realtime_server_keeps_heavy_session_work_off_event_loop():
    source = _source_text()

    assert "None if args.disable_spk else HybridSpeakerTracker" in source
    assert "await run_session_work(args, session.add_audio, message)" in source
    assert "await run_session_work(args, session.decode, is_final=True)" in source
    assert "await run_session_work(args, session.decode, is_final=False)" in source
