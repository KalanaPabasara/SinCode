from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import requests


LOCAL_FEEDBACK_PATH = Path(__file__).parent / "misc" / "feedback_submissions.jsonl"


class FeedbackStore:
    def __init__(
        self,
        supabase_url: str = "",
        supabase_anon_key: str = "",
        supabase_service_key: str = "",
        table_name: str = "feedback_submissions",
    ) -> None:
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_anon_key = supabase_anon_key
        self.supabase_service_key = supabase_service_key
        self.table_name = table_name

    @property
    def is_remote_enabled(self) -> bool:
        return bool(self.supabase_url and (self.supabase_service_key or self.supabase_anon_key))

    @property
    def backend_label(self) -> str:
        if self.is_remote_enabled:
            return "Supabase"
        return f"Local file ({LOCAL_FEEDBACK_PATH.name})"

    def save_submission(
        self,
        input_sentence: str,
        original_output: str,
        corrected_output: str,
        user_comment: str = "",
        decode_mode: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "input_sentence": input_sentence,
            "original_output": original_output,
            "corrected_output": corrected_output,
            "user_comment": user_comment.strip(),
            "decode_mode": decode_mode,
            "review_status": "pending",
            "admin_notes": "",
            "source": "streamlit",
        }
        if self.is_remote_enabled:
            return self._insert_remote(payload)
        return self._insert_local(payload)

    def _insert_local(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        LOCAL_FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOCAL_FEEDBACK_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"ok": True, "record": payload}

    def _insert_remote(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        response = requests.post(
            url,
            headers=self._headers(admin=False, prefer="return=representation"),
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json()
        row = rows[0] if rows else payload
        return {"ok": True, "record": row}

    def _headers(self, admin: bool, prefer: str = "") -> Dict[str, str]:
        key = self.supabase_service_key if admin and self.supabase_service_key else self.supabase_anon_key or self.supabase_service_key
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers


def format_feedback_error(exc: Exception) -> str:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                message = payload.get("message") or payload.get("hint") or json.dumps(payload)
                return f"{exc.response.status_code}: {message}"
        except ValueError:
            pass
        return f"{exc.response.status_code}: {exc.response.text.strip()}"
    return str(exc)
