import hashlib, json
from datetime import datetime

def make_version_id(params: dict, tag: str = None) -> str:
    """Create a compact version string based on timestamp + params hash."""
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    blob = json.dumps(params or {}, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha1(blob).hexdigest()[:8]
    base = f"v{stamp}-{h}"
    return f"{base}-{tag}" if tag else base
