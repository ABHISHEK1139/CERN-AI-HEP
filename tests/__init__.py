"""
Test suite for CERN-AI-HEP.

This is a real package (has ``__init__.py``) and ``pyproject.toml`` sets
``pythonpath = ["."]``. Both are required because an unrelated ``tests``
distribution is present in site-packages on some machines, which otherwise
shadows this directory during collection.
"""
