"""event_ingestion regression tests: config, features, statistics, synthetic, downloader."""

import json

import numpy as np
import pytest
import torch

from event_ingestion.config import (
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_PARTICLE_TYPES,
    PARTICLE_FEATURES,
    EventConfig,
)
from event_ingestion.statistics import EventStatistics
from event_ingestion.synthetic import SyntheticEventGenerator

# ==========================================================================
# EventConfig
# ==========================================================================

class TestEventConfig:
    def test_defaults_are_valid(self):
        EventConfig()

    @pytest.mark.parametrize("kwargs, msg", [
        ({"graph_strategy": "bogus"}, "graph_strategy"),
        ({"knn_k": 0}, "knn_k"),
        ({"delta_r_threshold": 0.0}, "delta_r_threshold"),
        ({"min_particles": 1}, "min_particles"),
        ({"max_particles": 1, "min_particles": 5}, "max_particles"),
        ({"particle_types": ["Bogus"]}, "Unknown particle_types"),
    ])
    def test_rejects_bad_config(self, kwargs, msg):
        with pytest.raises(ValueError, match=msg):
            EventConfig(**kwargs)

    def test_feature_dims_are_consistent(self):
        assert NUM_PARTICLE_TYPES == len(PARTICLE_FEATURES) == 5
        assert NODE_FEATURE_DIM == 11
        assert EDGE_FEATURE_DIM == 4

    def test_ensure_dirs(self, tmp_path):
        cfg = EventConfig(
            data_dir=tmp_path / "d",
            raw_dir=tmp_path / "d" / "raw",
            synthetic_dir=tmp_path / "d" / "syn",
            graph_dir=tmp_path / "d" / "g",
        )
        cfg.ensure_dirs()
        for p in (cfg.data_dir, cfg.raw_dir, cfg.synthetic_dir, cfg.graph_dir):
            assert p.is_dir()

    def test_node_types_are_unique(self):
        types = [info["node_type"] for info in PARTICLE_FEATURES.values()]
        assert len(set(types)) == len(types)


# ==========================================================================
# EventStatistics
# ==========================================================================

def _event(eid=0, n=3):
    return {
        "event_id": eid,
        "n_particles": n,
        "particles": [
            {
                "type": "Jet", "pt": 10.0 * (i + 1), "eta": 0.1 * i,
                "phi": 0.2 * i, "mass": 5.0, "energy": 10.0 * (i + 1),
            }
            for i in range(n)
        ],
    }


class TestEventStatistics:
    def test_empty_list(self):
        """An empty input returns the full zero-valued schema."""
        s = EventStatistics().compute([])
        assert s["n_events"] == 0
        assert s["n_particles_total"] == 0
        assert s["type_counts"] == {}
        # Same keys as a populated summary, so callers never need a guard.
        populated = EventStatistics().compute([_event()])
        assert set(s) == set(populated)

    def test_basic_summary(self):
        s = EventStatistics().compute([_event(i) for i in range(4)])
        assert s["n_events"] == 4
        assert s["n_particles_total"] == 12
        assert s["type_counts"]["Jet"] == 12
        assert s["pt_GeV"]["mean"] == pytest.approx(20.0)
        assert "mass_GeV" in s
        assert "energy_GeV" in s

    def test_zero_particle_events_keep_full_schema(self, capsys):
        """Regression: the zero-particle return omitted pt_GeV/eta/phi.

        print_summary then raised ``KeyError: 'pt_GeV'`` whenever every event
        had been filtered out.
        """
        events = [{"event_id": 0, "n_particles": 0, "particles": []}]
        summary = EventStatistics().compute(events)
        for key in ("pt_GeV", "eta", "phi", "mass_GeV", "energy_GeV",
                    "type_counts", "particles_per_event"):
            assert key in summary, f"missing key: {key}"
        # The full printing path must not raise.
        EventStatistics().print_summary(events)
        assert "No particles survived" in capsys.readouterr().out

    def test_particles_missing_required_keys_are_skipped(self):
        s = EventStatistics().compute([{"event_id": 0, "particles": [{"eta": 1.0}]}])
        assert s["n_particles_total"] == 0

    def test_print_summary_empty(self, capsys):
        EventStatistics().print_summary([])
        assert "No events" in capsys.readouterr().out

    def test_print_summary_populated(self, capsys):
        EventStatistics().print_summary([_event(i) for i in range(3)])
        out = capsys.readouterr().out
        assert "COLLISION EVENT STATISTICS" in out
        assert "Transverse momentum" in out

    def test_console_output_is_ascii_only(self, capsys):
        """Regression: "eta" and "+/-" as Unicode crashed the CLI.

        A default Windows console is cp1252, so ``print_summary`` raised
        ``UnicodeEncodeError`` and aborted ``python -m event_ingestion.synthetic``
        right after it had already written the dataset.
        """
        EventStatistics().print_summary([_event(i) for i in range(5)])
        out = capsys.readouterr().out
        offenders = {c for c in out if ord(c) > 127}
        assert not offenders, f"non-ASCII in console output: {offenders}"

    def test_print_summary_runs_under_cp1252(self, tmp_path):
        """Round-trip the output through a cp1252 encoder, as Windows would."""
        import io as _io

        buf = _io.StringIO()
        import contextlib

        with contextlib.redirect_stdout(buf):
            EventStatistics().print_summary([_event(i) for i in range(5)])
        text = buf.getvalue()
        text.encode("cp1252")  # raises if any character is unrepresentable

    def test_deterministic(self):
        events = [_event(i) for i in range(5)]
        a = EventStatistics().compute(events)
        b = EventStatistics().compute(events)
        assert a == b

    def test_plot_distributions_handles_empty(self, tmp_path):
        EventStatistics().plot_distributions([], output_dir=str(tmp_path))
        EventStatistics().plot_distributions(
            [{"event_id": 0, "particles": []}], output_dir=str(tmp_path)
        )
        assert not (tmp_path / "event_distributions.png").exists()

    def test_plot_distributions_writes_figure(self, tmp_path):
        EventStatistics().plot_distributions(
            [_event(i) for i in range(20)], output_dir=str(tmp_path)
        )
        assert (tmp_path / "event_distributions.png").exists()


# ==========================================================================
# SyntheticEventGenerator
# ==========================================================================

class TestSyntheticGenerator:
    def test_counts_and_labels(self):
        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(n_normal=20, n_anomaly=5)
        assert len(events) == 25
        assert labels.sum() == 5
        assert set(labels.tolist()) == {0, 1}
        assert len(labels) == len(events)

    def test_reproducible(self):
        a, la = SyntheticEventGenerator(seed=3).generate(10, 4)
        b, lb = SyntheticEventGenerator(seed=3).generate(10, 4)
        assert la.tolist() == lb.tolist()
        assert [e["event_id"] for e in a] == [e["event_id"] for e in b]

    @pytest.mark.parametrize("n_norm, n_anom", [(-1, 5), (5, -1), (0, 0)])
    def test_rejects_bad_counts(self, n_norm, n_anom):
        with pytest.raises(ValueError):
            SyntheticEventGenerator(seed=1).generate(n_norm, n_anom)

    def test_particle_energy_is_physically_consistent(self):
        """E^2 = pT^2 cosh^2(eta) + m^2 for every generated particle."""
        gen = SyntheticEventGenerator(seed=5)
        for _ in range(200):
            ev = gen._generate_event(0, anomalous=True)
            for p in ev["particles"]:
                if p["type"] == "MET":
                    continue
                expected = np.sqrt(
                    p["pt"] ** 2 * np.cosh(p["eta"]) ** 2 + p["mass"] ** 2
                )
                assert np.isclose(p["energy"], expected, rtol=1e-6), p

    def test_energy_eta_overflow_is_bounded(self):
        assert np.isfinite(SyntheticEventGenerator._energy(100.0, 1e6, 10.0))
        assert np.isfinite(SyntheticEventGenerator._energy(100.0, -1e6, 10.0))

    def test_particle_count_within_config_bounds(self):
        cfg = EventConfig(min_particles=3, max_particles=20)
        gen = SyntheticEventGenerator(config=cfg, seed=2)
        for _ in range(100):
            ev = gen._generate_event(0, anomalous=True)
            assert cfg.min_particles <= ev["n_particles"] <= cfg.max_particles

    def test_at_most_one_met_per_event(self):
        gen = SyntheticEventGenerator(seed=9)
        for _ in range(100):
            ev = gen._generate_event(0)
            assert sum(p["type"] == "MET" for p in ev["particles"]) <= 1

    def test_all_values_finite(self):
        gen = SyntheticEventGenerator(seed=11)
        events, _ = gen.generate(50, 20)
        for ev in events:
            for p in ev["particles"]:
                for k, v in p.items():
                    if isinstance(v, float):
                        assert np.isfinite(v), f"{k}={v} in {p}"

    def test_save_returns_existing_path_without_suffix(self, tmp_path):
        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(5, 2)
        # No .npz on the given name: numpy appends it, so the returned path
        # must reflect that or callers get a path that does not exist.
        out = gen.save(tmp_path / "events", events, labels)
        assert out.exists()
        assert out.suffix == ".npz"

    def test_save_with_explicit_suffix(self, tmp_path):
        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(5, 2)
        out = gen.save(tmp_path / "events.npz", events, labels)
        assert out.exists()

    def test_save_roundtrips(self, tmp_path):
        from event_ingestion.loader import EventLoader

        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(8, 3)
        path = gen.save(tmp_path / "e.npz", events, labels)
        loaded = EventLoader().load_synthetic(path)
        assert len(loaded) == 11

    def test_save_rejects_length_mismatch(self, tmp_path):
        gen = SyntheticEventGenerator(seed=1)
        events, labels = gen.generate(5, 2)
        with pytest.raises(ValueError, match="length mismatch"):
            gen.save(tmp_path / "e.npz", events, labels[:3])


# ==========================================================================
# CERN downloader
# ==========================================================================

# Shape the CERN Open Data portal actually returns: `files` is a dict whose
# `value` list holds per-file entries, each with a bare-filename `key` and a
# `links.self` API path.
PORTAL_RECORD = {
    "id": 12353,
    "metadata": {"access_right": "open", "title": "DYJets"},
    "files": {
        "entry": 1,
        "description": "Files",
        "value": [
            {
                "key": "DYJetsToLL.root",
                "size": 1024,
                "links": {
                    "self": "/api/records/12353/files/DYJetsToLL.root",
                    "content": "https://opendata.cern.ch/api/records/12353/files/DYJetsToLL.root/content",
                },
            },
            {
                "key": "README.txt",
                "size": 10,
                "links": {"self": "/api/records/12353/files/README.txt"},
            },
        ],
    },
}


@pytest.fixture
def downloader(tmp_path):
    from event_ingestion.downloader import CMSDataDownloader

    cfg = EventConfig(
        data_dir=tmp_path / "data",
        raw_dir=tmp_path / "data" / "raw",
        synthetic_dir=tmp_path / "data" / "syn",
        graph_dir=tmp_path / "data" / "g",
    )
    return CMSDataDownloader(cfg)


def _stub_urlopen(monkeypatch, payload):
    import io
    import urllib.request

    class Ctx:
        def __init__(self):
            self._buf = io.BytesIO(json.dumps(payload).encode())

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self._buf.getvalue()

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: Ctx())


class TestDownloaderRecordParsing:
    def test_finds_root_in_portal_shape(self, downloader, monkeypatch):
        """Regression: only a *list* of dicts was inspected.

        The portal returns a dict, so every lookup produced an empty list and
        every download reported "no ROOT files found".
        """
        _stub_urlopen(monkeypatch, PORTAL_RECORD)
        files = downloader.get_record_files(12353)
        assert len(files) == 1
        assert files[0]["uri"] == "/api/records/12353/files/DYJetsToLL.root"
        assert files[0]["size"] == 1024

    def test_uses_uri_not_bare_filename_key(self, downloader, monkeypatch):
        """`key` is a filename, not a URI.

        Concatenating it onto the host would yield
        "https://opendata.cern.chDYJetsToLL.root", so the parser must return
        the `links.self` path instead.
        """
        _stub_urlopen(monkeypatch, PORTAL_RECORD)
        uri = downloader.get_record_files(12353)[0]["uri"]
        assert uri.startswith("/") or uri.startswith("http")
        assert uri != "DYJetsToLL.root"
        # The URL that download_file would build must be a valid absolute URL.
        url = uri if uri.startswith("http") else f"https://opendata.cern.ch{uri}"
        assert url.startswith("https://opendata.cern.ch/api/")

    def test_handles_flat_list_shape(self, downloader, monkeypatch):
        _stub_urlopen(monkeypatch, {
            "files": [
                {"uri": "/api/records/1/files/a.root", "size": 5},
                {"uri": "/api/records/1/files/b.root", "size": 6},
            ]
        })
        assert len(downloader.get_record_files(1)) == 2

    def test_deduplicates(self, downloader, monkeypatch):
        _stub_urlopen(monkeypatch, {
            "files": [
                {"uri": "/x/a.root", "size": 1},
                {"uri": "/x/a.root", "size": 1},
            ]
        })
        assert len(downloader.get_record_files(1)) == 1

    def test_no_root_files_returns_empty(self, downloader, monkeypatch):
        _stub_urlopen(monkeypatch, {"files": []})
        assert downloader.get_record_files(1) == []

    def test_non_dict_payload_raises(self, downloader, monkeypatch):
        _stub_urlopen(monkeypatch, [1, 2, 3])
        with pytest.raises(ValueError, match="JSON object"):
            downloader.get_record_files(1)

    def test_recommended_datasets_registry(self, downloader):
        available = downloader.list_available()
        assert "doublemuon_2012" in available
        for key, info in available.items():
            assert isinstance(info["record_id"], int)
            assert info["description"]


class TestDownloaderFileTransfer:
    def test_rejects_bad_max_size(self, downloader, tmp_path):
        with pytest.raises(ValueError, match="max_size_mb"):
            downloader.download_file("/x/a.root", tmp_path, max_size_mb=0)

    def test_reuses_existing_nonempty_file(self, downloader, tmp_path):
        target = tmp_path / "a.root"
        target.write_bytes(b"x" * 10)
        assert downloader.download_file("/x/a.root", tmp_path) == target

    def test_replaces_zero_byte_file(self, downloader, tmp_path, monkeypatch):
        target = tmp_path / "a.root"
        target.write_bytes(b"")
        import io
        import urllib.request

        class Resp:
            headers = {"Content-Length": "3"}

            def read(self, n):
                return self._buf.read(n)

            def __enter__(self):
                self._buf = io.BytesIO(b"abc")
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=None: Resp())
        out = downloader.download_file("/x/a.root", tmp_path)
        assert out.read_bytes() == b"abc"

    def test_incomplete_download_raises_and_leaves_no_partial(self, downloader,
                                                             tmp_path, monkeypatch):
        """A truncated transfer must not be published under the real name."""
        import io
        import urllib.request

        class Resp:
            headers = {"Content-Length": "100"}

            def read(self, n):
                return self._buf.read(n)

            def __enter__(self):
                self._buf = io.BytesIO(b"only-10-b")
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=None: Resp())
        with pytest.raises(ValueError, match="Incomplete download"):
            downloader.download_file("/x/big.root", tmp_path)
        assert not (tmp_path / "big.root").exists()
        assert not (tmp_path / "big.root.partial").exists()

    def test_size_limit_enforced_mid_stream(self, downloader, tmp_path,
                                             monkeypatch):
        import io
        import urllib.request

        class Resp:
            headers = {}  # no Content-Length: only the streaming cap can catch it

            def read(self, n):
                return self._buf.read(n)

            def __enter__(self):
                self._buf = io.BytesIO(b"y" * (3 << 20))
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=None: Resp())
        with pytest.raises(ValueError, match="exceeded"):
            downloader.download_file("/x/huge.root", tmp_path, max_size_mb=1)

    def test_empty_response_raises(self, downloader, tmp_path, monkeypatch):
        import io
        import urllib.request

        class Resp:
            headers = {}

            def read(self, n):
                return b""

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout=None: Resp())
        with pytest.raises(ValueError, match="empty file"):
            downloader.download_file("/x/empty.root", tmp_path)
