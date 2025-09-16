
import io
import yaml

def load_config(path: str):
    """
    Load a YAML config with UTF-8 (BOM-tolerant) so Windows codepages don't break.
    """
    # Read as UTF-8, but tolerate BOM if present
    with io.open(path, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()

    # Optional: normalize weird line endings just in case
    if "\r\n" in text:
        text = text.replace("\r\n", "\n")

    return yaml.safe_load(text) or {}
