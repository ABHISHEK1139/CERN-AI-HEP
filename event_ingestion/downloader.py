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

# Curated small NanoAOD samples suitable for ML development.
# Each entry: (record_id, description, approximate_size_mb)
RECOMMENDED_DATASETS = {
    "doublemuon_2012": {
        "record_id": 6021,
        "description": "CMS DoubleMuParked 2012B — dimuon events, NanoAOD format",
        "files_pattern": "*.root",
        "approx_size_mb": 50,
    },
    "singlemuon_2015": {
        "record_id": 24119,
        "description": "CMS SingleMuon 2015D — single muon trigger, NanoAODRun2",
        "files_pattern": "*.root",
        "approx_size_mb": 200,
    },
    "ttbar_mc": {
        "record_id": 19980,
        "description": "TTbar Monte Carlo simulation — NanoAOD (good for ML benchmarks)",
        "files_pattern": "*.root",
        "approx_size_mb": 100,
    },
    "higgs_mc": {
        "record_id": 12361,
        "description": "Higgs to 4 leptons MC — NanoAODSIM (classic analysis channel)",
        "files_pattern": "*.root",
        "approx_size_mb": 30,
    },
}

DEFAULT_DATASET = "doublemuon_2012"


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

        The portal's record endpoint returns several different shapes, so all of
        them are handled here. The old implementation only looked for a *list*
        of dicts under ``metadata.files``; the portal actually returns a
        *dict* of the form
        ``{"entry": 1, "description": ..., "value": [{"key": ..., "size": ...,
        "links": {"self": "/api/records/1234/files/NAME"}}]}``, which meant
        every lookup silently produced an empty list and every download
        reported "no ROOT files found".

        Args:
            record_id: CERN Open Data record identifier.

        Returns:
            List of dicts with 'uri' and 'size' keys. Empty if the record
            exposes no downloadable ROOT files.

        Raises:
            ValueError: If the record metadata cannot be parsed.
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

        files: list[dict[str, str]] = []
        seen = set()

        def _add(uri: str | None, size: Any = 0) -> None:
            """Record one ROOT file entry.

            Only *URIs* are accepted. The portal also exposes a ``key`` field
            holding a bare filename ("DYJetsToLL.root"); feeding that to
            :meth:`download_file` would concatenate it onto the host and build
            "https://opendata.cern.chDYJetsToLL.root". Real URIs start with a
            scheme or a leading slash, so anything else is rejected.
            """
            if not isinstance(uri, str) or not uri:
                return
            uri = uri.strip()
            if not (uri.startswith("http://") or uri.startswith("https://") or uri.startswith("/")):
                return
            if not uri.lower().split("?")[0].endswith(".root"):
                return
            if uri in seen:
                return
            seen.add(uri)
            try:
                size_int = int(size) if size else 0
            except (TypeError, ValueError):
                size_int = 0
            files.append({"uri": uri, "size": size_int})

        def _walk(node: Any) -> None:
            """Depth-first scan for anything that looks like a ROOT file entry."""
            if isinstance(node, dict):
                # 'links'/'self' is how the portal addresses individual files and
                # is preferred over 'key', which is only a filename.
                links = node.get("links")
                if isinstance(links, dict) and links.get("self"):
                    _add(links["self"], node.get("size", node.get("bytes", 0)))
                _add(node.get("uri"), node.get("size", node.get("bytes", 0)))
                for value in node.values():
                    if isinstance(value, (dict, list)):
                        _walk(value)
            elif isinstance(node, list):
                for value in node:
                    if isinstance(value, (dict, list)):
                        _walk(value)

        _walk(data)

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
