"""PDF compression helper (optional PyMuPDF dependency)."""
import os

import pytest

# `fitz` is the legacy import name; canonical package is now `pymupdf`.
pymupdf = pytest.importorskip("pymupdf", reason="PyMuPDF not installed (optional)")


def compress_pdf(input_path, output_path):
    print(f"Opening {input_path}...")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input PDF not found: {input_path}")
    doc = pymupdf.open(input_path)

    # Save with compression options
    # garbage=4: complete garbage collection and duplicate object merging
    # deflate=True: compress streams
    print("Saving compressed PDF...")
    doc.save(output_path, garbage=4, deflate=True, clean=True)

    orig_size = os.path.getsize(input_path)
    new_size = os.path.getsize(output_path)
    print(f"Original size: {orig_size / 1024 / 1024:.2f} MB")
    print(f"Compressed size: {new_size / 1024 / 1024:.2f} MB")
    print(f"Reduction: {(1 - new_size / orig_size) * 100:.2f}%")


def test_compress_helper_importable():
    assert callable(compress_pdf)


if __name__ == "__main__":
    compress_pdf("final_report.pdf", "final_report_compressed.pdf")
