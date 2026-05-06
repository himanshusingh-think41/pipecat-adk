"""Shared constants and utilities for the [HEARD] event format."""

import re

# Format: <system>[HEARD] invocation_id="{id}" Candidate only heard: "{text}"</system>
HEARD_FORMAT = (
    '<system>[HEARD] invocation_id="{invocation_id}" '
    'Candidate only heard: "{heard_text}"</system>'
)

HEARD_PATTERN = re.compile(
    r'\[HEARD\]\s*invocation_id="([^"]*)"\s*Candidate only heard:\s*"(.*?)"',
    re.DOTALL,
)
