"""GEMQ package initialization."""

import os
from functools import wraps


def _patch_huggingface_pagination() -> None:
    """Rewrite Hugging Face pagination links to the configured mirror."""
    endpoint = os.environ.get("HF_ENDPOINT", "").rstrip("/")
    official_endpoint = "https://huggingface.co"

    if not endpoint or endpoint == official_endpoint:
        return

    import huggingface_hub.utils._pagination as pagination

    original = getattr(pagination, "_get_next_page", None)
    if original is None or getattr(original, "_gemq_mirror_patch", False):
        return

    @wraps(original)
    def mirror_get_next_page(response):
        url = original(response)
        if url and url.startswith(official_endpoint):
            return endpoint + url[len(official_endpoint) :]
        return url

    mirror_get_next_page._gemq_mirror_patch = True
    pagination._get_next_page = mirror_get_next_page


_patch_huggingface_pagination()
