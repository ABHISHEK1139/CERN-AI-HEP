"""
CMS Open Data downloader.

Downloads small NanoAOD samples from the CERN Open Data portal
via HTTP. Supports both real CMS collision data and Monte Carlo
simulation samples.

Usage:
    from event_ingestion.downloader import CMSDataDownloader
    downloader = CMSDataDownloader()
    downloader.download()

Or from CLI:
    python -m event_ingestion.downloader
"""

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from event_ingestion.config import EventConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CMS Open Data Portal Endpoints
# ----------------------------------------------------------------------------

CERN_OPENDATA_API = "https://opendata.cern.ch/api/records"

REQUEST_TIMEOUT = 30      # seconds, for record metadata lookups
DOWNLOAD_TIMEOUT = 300    # seconds, per socket read during file transfer
CHUNK_SIZE = 1 << 20      # 1 MiB read granularity

# Curated NanoAOD samples suitable for ML development.
# Each entry: (record_id, description, approximate_size_mb)
#
# Record IDs below were verified against the live portal: every one of these
# records currently exposes a `files` list, and the .root entries resolve.
# (`higgs_mc` was previously 12361, which does not contain the expected file;
# the correct record for GluGluToHToTauTau.root is 12351.)
RECOMMENDED_DATASETS = {
    "dyjets": {
        "record_id": 12353,
        "description": "CMS DYJetsToLL - inclusive QCD dijet sample (NanoAOD)",
        "files_pattern": "*.root",
        "approx_size_mb": 8600,
        "expected_path": "data/cms/dyjets/DYJetsToLL.root",
    },
    "ttbar_mc": {
        "record_id": 12354,
        "description": "CMS TTbar Monte Carlo - top-pair sample (NanoAOD)",
        "files_pattern": "*.root",
        "approx_size_mb": 3300,
        "expected_path": "data/cms/ttbar/TTbar.root",
    },
    "higgs_mc": {
        "record_id": 12351,
        "description": "CMS GluGluToHToTauTau Monte Carlo (reduced NanoAOD, for education)",
        "files_pattern": "*.root",
        "approx_size_mb": 190,
        "expected_path": "data/cms/higgs/GluGluToHToTauTau.root",
    },
    "doublemuon_2012": {
        "record_id": 6021,
        "description": "CMS DoubleMuParked 2012B - dimuon events, NanoAOD format",
        "files_pattern": "*.root",
        "approx_size_mb": 50,
    },
    "singlemuon_2015": {
        "record_id": 24119,
        "description": "CMS SingleMuon 2015D - single muon trigger, NanoAODRun2",
        "files_pattern": "*.root",
        "approx_size_mb": 200,
    },
}

DEFAULT_DATASET = "higgs_mc"


def _root_uri_to_http(uri: str) -> str:
    """Convert an EOS ``root://`` URI to its opendata.cern.ch HTTP equivalent.

    The portal serves files as ``root://eospublic.cern.ch//eos/opendata/...``.
    The equivalent download URL is
    ``https://opendata.cern.ch/eos/opendata/...`` (note the single slash after
    the host: the ``//`` in the root URI is path-root, not a URL authority).

    Args:
        uri: Either a ``root://`` URI or an already-HTTP URL.

    Returns:
        An absolute https URL, or the input unchanged if no mapping applies.
    """
    if not uri.startswith("root://"):
        return uri
    host_and_path = uri[len("root://"):]
    if "/" not in host_and_path:
        return uri
    path = host_and_path.split("/", 1)[1].lstrip("/")
    return f"https://opendata.cern.ch/{path}"


class CMSDataDownloader:
    """Download CMS Open Data NanoAOD files from CERN portal."""

    def __init__(self, config: EventConfig | None = None):
        self.config = config or EventConfig()
        self.config.ensure_dirs()

    def list_available(self) -> dict[str, dict]:
        """List recommended datasets."""
        return RECOMMENDED_DATASETS

    def get_record_files(self, record_id: int) -> list[dict[str, str]]:
        """
        Fetch the file list for a CERN Open Data record.

        Verified against the live portal: the response is
        ``{"metadata": {"files": [{"key", "size", "uri": "root://..."}],
        "_files": [...], "_file_indices": []}}``. ``uri`` is an EOS
        ``root://`` reference, **not** an HTTP path, so it is translated by
        :func:`_root_uri_to_http` before being returned. ``_files`` and
        ``_file_indices`` are accepted as fallbacks because the shape varies
        between record generations.

        Args:
            record_id: CERN Open Data record identifier.

        Returns:
            List of dicts with ``uri`` (absolute https), ``key`` and ``size``.

        Raises:
            ValueError: If the metadata cannot be parsed.
        """
        url = f"{CERN_OPENDATA_API}/{record_id}"
        logger.info("Fetching record metadata from %s", url)

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            logger.error("Failed to fetch record %s: %s", record_id, e)
            raise
        except json.JSONDecodeError as e:
            raise ValueError(f"Record {record_id} returned non-JSON metadata: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected a JSON object for record {record_id}, got {type(data).__name__}."
            )

        metadata = data.get("metadata", data)
        files: list[dict[str, str]] = []
        seen = set()

        def _add(uri: str | None, key: str | None, size: Any) -> None:
            if not isinstance(uri, str) or not uri:
                return
            http = _root_uri_to_http(uri)
            if not http.startswith(("http://", "https://")):
                return
            filename = (key or http.rsplit("/", 1)[-1]).split("?")[0]
            if not filename.lower().endswith(".root"):
                return
            if http in seen:
                return
            seen.add(http)
            try:
                size_int = int(size) if size else 0
            except (TypeError, ValueError):
                size_int = 0
            files.append({"uri": http, "key": filename, "size": size_int})

        # Primary location first, then the variants older records use.
        for field_name in ("files", "_files", "file_indices", "_file_indices"):
            entries = metadata.get(field_name) or []
            if isinstance(entries, dict):
                entries = entries.get("value", [])
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                # A record can expose the same file under both `uri` and
                # links.self; _add dedupes on the resolved URL.
                _add(entry.get("uri"), entry.get("key"), entry.get("size", 0))
                links = entry.get("links")
                if isinstance(links, dict):
                    _add(links.get("self"), entry.get("key"), entry.get("size", 0))

        logger.info("Found %d ROOT file(s) for record %s", len(files), record_id)
        if not files:
            logger.warning(
                "No .root entries in the metadata for record %s. Visit "
                "https://opendata.cern.ch/record/%s to inspect it manually.",
                record_id, record_id,
            )
        return files

    def download_file(self, uri: str, output_dir: Path, max_size_mb: int = 500) -> Path:
        """
        Download a single file from CERN Open Data.

        The transfer writes to a ``.partial`` sibling and is moved into place
        only after the byte count matches ``Content-Length``. A previous
        implementation streamed straight to the final path, so any interruption
        left a truncated file that the next run happily accepted as complete
        (it only checked ``size > 0``).

        Args:
            uri: File URI (relative or absolute).
            output_dir: Directory to save the file.
            max_size_mb: Maximum file size to download (safety limit).

        Returns:
            Path to the downloaded file.

        Raises:
            ValueError: If the file exceeds ``max_size_mb`` or the download is
                incomplete.
        """
        if max_size_mb < 1:
            raise ValueError(f"max_size_mb must be >= 1, got {max_size_mb}.")

        # Construct full URL
        if uri.startswith("http"):
            url = uri
        else:
            url = f"https://opendata.cern.ch{uri}"

        filename = Path(uri.split("?")[0]).name or "download.root"
        # Sanitize Windows-illegal characters from query-derived names
        filename = "".join(
            c for c in filename if c not in '<>:"/\\|?*'
        ).strip() or "download.root"
        output_path = output_dir / filename

        if output_path.exists():
            size = output_path.stat().st_size
            if size > 0:
                logger.info("File already exists: %s (%d bytes)", output_path, size)
                return output_path
            logger.warning("Removing zero-byte partial download: %s", output_path)
            output_path.unlink()

        logger.info("Downloading %s -> %s", url, output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_path = output_path.with_suffix(output_path.suffix + ".partial")

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT) as resp:
                # Check content length
                content_length = resp.headers.get("Content-Length")
                expected = 0
                if content_length is not None:
                    try:
                        expected = int(content_length)
                    except (TypeError, ValueError):
                        expected = 0
                if expected and expected > max_size_mb * 1024 * 1024:
                    raise ValueError(
                        f"File too large: {expected / 1024 / 1024:.0f} MB "
                        f"(limit: {max_size_mb} MB). Use max_size_mb to increase."
                    )

                # Stream download
                downloaded = 0
                with open(partial_path, "wb") as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if max_size_mb and downloaded > max_size_mb * 1024 * 1024:
                            raise ValueError(
                                f"Transfer exceeded the {max_size_mb} MB limit "
                                f"({downloaded / 1024 / 1024:.0f} MB) for {filename}."
                            )

            if expected and downloaded != expected:
                raise ValueError(
                    f"Incomplete download of {filename}: got {downloaded} bytes, "
                    f"expected {expected}. Re-run to retry."
                )
            if downloaded == 0:
                raise ValueError(f"Download of {filename} produced an empty file.")

            # Atomic publish: only now does the file appear under its real name.
            partial_path.replace(output_path)
        except Exception as e:
            logger.error("Download failed: %s", e)
            if partial_path.exists():
                try:
                    partial_path.unlink()
                except OSError:
                    pass
            raise

        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info("Downloaded %s (%.1f MB)", filename, size_mb)
        return output_path

    def download(
        self,
        dataset_key: str = DEFAULT_DATASET,
        max_files: int = 1,
        max_size_mb: int = 500,
    ) -> list[Path]:
        """
        Download a recommended CMS dataset.

        Args:
            dataset_key: Key from RECOMMENDED_DATASETS.
            max_files: Maximum number of ROOT files to download.
            max_size_mb: Maximum file size per file.

        Returns:
            List of paths to downloaded ROOT files.
        """
        if dataset_key not in RECOMMENDED_DATASETS:
            available = ", ".join(RECOMMENDED_DATASETS.keys())
            raise ValueError(f"Unknown dataset '{dataset_key}'. Available: {available}")

        dataset = RECOMMENDED_DATASETS[dataset_key]
        record_id = dataset["record_id"]

        logger.info(f"Dataset: {dataset['description']}")
        logger.info(f"Record ID: {record_id}")

        # Fetch file list
        files = self.get_record_files(record_id)

        if not files:
            logger.warning(
                f"No ROOT files found via API for record {record_id}. "
                f"Visit https://opendata.cern.ch/record/{record_id} to download manually."
            )
            return []

        # Download up to max_files
        downloaded = []
        for file_info in files[:max_files]:
            uri = file_info.get("uri", "")
            if not uri:
                logger.warning(f"Skipping record entry without URI: {file_info}")
                continue
            path = self.download_file(
                uri,
                self.config.raw_dir / dataset_key,
                max_size_mb=max_size_mb,
            )
            downloaded.append(path)

        logger.info(f"Downloaded {len(downloaded)} file(s) to {self.config.raw_dir / dataset_key}")
        return downloaded


def main():
    """CLI entry point for downloading CMS data."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Download CMS Open Data")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=list(RECOMMENDED_DATASETS.keys()),
        help=f"Dataset to download (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--max-files", type=int, default=1,
        help="Maximum number of ROOT files to download",
    )
    parser.add_argument(
        "--max-size-mb", type=int, default=500,
        help="Maximum file size in MB",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_datasets",
        help="List available datasets",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("\nAvailable CMS Open Data samples:\n")
        for key, info in RECOMMENDED_DATASETS.items():
            print(f"  {key:20s}  {info['description']}")
            print(f"  {'':20s}  ~{info['approx_size_mb']} MB, record #{info['record_id']}")
            print()
        return

    downloader = CMSDataDownloader()
    paths = downloader.download(
        dataset_key=args.dataset,
        max_files=args.max_files,
        max_size_mb=args.max_size_mb,
    )

    if paths:
        print(f"\nDownloaded {len(paths)} file(s):")
        for p in paths:
            print(f"  {p}")
    else:
        print("\nNo files downloaded. Check logs for details.")


if __name__ == "__main__":
    main()
