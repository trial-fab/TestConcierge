import os
from functools import lru_cache

@lru_cache(maxsize=1)
def get_splunk_settings() -> dict[str, object]:
    batch_size = int(os.getenv("SPLUNK_BATCH_SIZE", "10"))
    flush_interval = float(os.getenv("SPLUNK_FLUSH_INTERVAL", "5"))
    hec_url = os.getenv("SPLUNK_HEC_URL", "")
    hec_token = os.getenv("SPLUNK_HEC_TOKEN", "")
    enabled = os.getenv("SPLUNK_ENABLED", "true").lower() in ("true", "1", "yes")
    if not hec_url or not hec_token:
        enabled = False
    return {
        "hec_url": hec_url,
        "hec_token": hec_token,
        "index": os.getenv("SPLUNK_INDEX", "main"),
        "sourcetype_prefix": os.getenv("SPLUNK_SOURCETYPE_PREFIX", "usf_concierge"),
        "batch_size": batch_size,
        "flush_interval": flush_interval,
        "enabled": enabled,
    }
