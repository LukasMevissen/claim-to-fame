"""QuantCo API client.

NOT WIRED UP YET: the endpoints and payload format are defined in API_HANDBOOK.md,
which is in the organisers' shared Drive folder. Once it is in vendor/quantco/:
1. fill [api] in configs/default.toml,
2. adapt fetch_key() and submit() to the handbook (port the calls from starter_script.py),
3. run `python -m c2f live --case 0 --dry-run` and then without --dry-run on case 0.
"""

from __future__ import annotations

import os

from .config import Config


class ApiNotConfigured(RuntimeError):
    pass


class QuantCoClient:
    def __init__(self, cfg: Config):
        api = cfg.section("api")
        if not api.get("base_url") or not api.get("key_endpoint") or not api.get("submit_endpoint"):
            raise ApiNotConfigured("Fill [api] in the config from API_HANDBOOK.md first.")
        self.base = api["base_url"].rstrip("/")
        self.key_ep = api["key_endpoint"]
        self.submit_ep = api["submit_endpoint"]
        self.api_key = os.environ["TEAM_API_KEY"]

    def _headers(self) -> dict:
        # TODO(handbook): confirm auth header name/format.
        return {"Authorization": f"Bearer {self.api_key}"}

    def fetch_key(self, case_id: str) -> str:
        import requests

        # TODO(handbook): confirm path/params and response field.
        r = requests.get(f"{self.base}{self.key_ep}".format(case_id=case_id),
                         headers=self._headers(), timeout=5)
        r.raise_for_status()
        return r.json()["key"]

    def submit(self, case_id: str, payload: list[dict]) -> dict:
        import requests

        # TODO(handbook): map payload field names (item id, a, b) to the API schema.
        r = requests.post(f"{self.base}{self.submit_ep}".format(case_id=case_id),
                          headers=self._headers(), json={"items": payload}, timeout=8)
        r.raise_for_status()
        return r.json() if r.content else {}
