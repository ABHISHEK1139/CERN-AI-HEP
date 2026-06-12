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
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional

from event_ingestion.config import EventConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# CMS Open Data Portal Endpoints
# ----------------------------------------------------------------------------

CERN_OPENDATA_API = "https://opendata.cern.ch/api/records"

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

    def __init__(self, config: Optional[EventConfig] = None):
        self.config = config or EventConfig()
        self.config.ensure_dirs()

    def list_available(self) -> Dict[str, dict]:
        """List recommended datasets."""
        return RECOMMENDED_DATASETS

    def get_record_files(self, record_id: int) -> List[Dict[str, str]]:
        """
        Fetch file list for a CERN Open Data record.

        Args:
            record_id: CERN Open Data record identifier.

        Returns:
            List of dicts with 'uri' and 'size' keys.
        """
        url = f"{CERN_OPENDATA_API}/{record_id}"
        logger.info(f"Fetching record metadata from {url}")

        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            logger.error(f"Failed to fetch record {record_id}: {e}")
            raise

        files = []
        # CERN Open Data API nests files under metadata.files or metadata.file_indices
        metadata = data.get("metadata", data)

        for file_entry in metadata.get("files", []):
            uri = file_entry.get("uri", "")
            size = file_entry.get("size", 0)
            if uri.endswith(".root"):
                files.append({"uri": uri, "size": size})

        # Fallback: check file_indices
        if not files:
            for idx in metadata.get("file_indices", []):
                uri = idx.get("uri", "")
                if uri:
                    files.append({"uri": uri, "size": 0})

        logger.info(f"Found {len(files)} ROOT file(s) for record {record_id}")
        return files

    def download_file(self, uri: str, output_dir: Path, max_size_mb: int = 500) -> Path:
        """
        Download a single file from CERN Open Data.

        Args:
            uri: File URI (relative or absolute).
            output_dir: Directory to save the file.
            max_size_mb: Maximum file size to download (safety limit).

        Returns:
            Path to downloaded file.
        """
        # Construct full URL
        if uri.startswith("http"):
            url = uri
        else:
            url = f"https://opendata.cern.ch{uri}"

        filename = Path(uri).name
        output_path = output_dir / filename

        if output_path.exists():
            logger.info(f"File already exists: {output_path}")
            return output_path

        logger.info(f"Downloading {url} -> {output_path}")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=300) as resp:
                # Check content length
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > max_size_mb * 1024 * 1024:
                    raise ValueError(
                        f"File too large: {int(content_length) / 1024 / 1024:.0f} MB "
                        f"(limit: {max_size_mb} MB). Use max_size_mb to increase."
                    )

                # Stream download
                with open(output_path, "wb") as f:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)

        except urllib.error.URLError as e:
            logger.error(f"Download failed: {e}")
            if output_path.exists():
                output_path.unlink()
            raise

        size_mb = output_path.stat().st_size / 1024 / 1024
        logger.info(f"Downloaded {filename} ({size_mb:.1f} MB)")
        return output_path

    def download(
        self,
        dataset_key: str = DEFAULT_DATASET,
        max_files: int = 1,
        max_size_mb: int = 500,
    ) -> List[Path]:
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
            path = self.download_file(
                file_info["uri"],
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
