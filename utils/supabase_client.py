import os
from functools import lru_cache

from supabase import Client, create_client

class SupabaseConfigError(RuntimeError):
    """Raised when the Supabase client cannot be initialised due to missing settings."""

@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_API_KEY")
        or os.getenv("SUPABASE_KEY")
    )

    if not url or not key:
        raise SupabaseConfigError(
            "Supabase configuration missing. Ensure SUPABASE_URL and "
            "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_API_KEY) are set."
        )

    return create_client(url, key)