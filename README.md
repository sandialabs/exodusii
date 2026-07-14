# exodusii

A modern Python 3.13+ interface for Exodus II finite element databases.

This is a refresh of the long-lived `exodusii` package. The new implementation uses:

- `src/` package layout
- Python 3.13+
- `dataclasses`
- `StrEnum`
- type hints
- `netCDF4`
- NumPy 2+
- pytest

The public API will remain string-friendly while the internals use typed models and enums.

## Development

Install in editable mode with test dependencies:

```bash
python -m pip install -e ".[test]"
```

Run tests

```bash
pytest
```
