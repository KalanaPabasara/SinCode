import json
from typing import Any, Dict, List, Optional

import requests

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
        return "Supabase" if self.is_remote_enabled else "Supabase (not configured)"

    def save_submission(
        self,
        input_sentence: str,
        original_output: str,
        corrected_output: str,
        user_comment: str = "",
        decode_mode: str = "",
    ) -> Dict[str, Any]:
        payload = {
            "input_sentence": input_sentence,
            "original_output": original_output,
            "corrected_output": corrected_output,
            "user_comment": user_comment.strip(),
            "decode_mode": decode_mode,
            "review_status": "pending",
            "admin_notes": "",
            "source": "streamlit",
        }

        self._require_remote()
        return self._insert_remote(payload)

    def list_submissions(self, review_status: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        self._require_remote()
        return self._list_remote(review_status=review_status, limit=limit)

    def update_submission_status(self, submission_id: str, review_status: str, admin_notes: str = "") -> Dict[str, Any]:
        self._require_remote()
        return self._update_remote(submission_id=submission_id, review_status=review_status, admin_notes=admin_notes)

    def _require_remote(self) -> None:
        if not self.is_remote_enabled:
            raise RuntimeError(
                "Supabase is not configured. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in secrets."
            )

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

    def _list_remote(self, review_status: Optional[str], limit: int) -> List[Dict[str, Any]]:
        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        params = {
            "select": "id,created_at,input_sentence,original_output,corrected_output,user_comment,decode_mode,review_status,admin_notes,source",
            "order": "created_at.desc",
            "limit": str(limit),
        }
        if review_status and review_status != "all":
            params["review_status"] = f"eq.{review_status}"

        response = requests.get(url, headers=self._headers(admin=True), params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    def _update_remote(self, submission_id: str, review_status: str, admin_notes: str) -> Dict[str, Any]:
        url = f"{self.supabase_url}/rest/v1/{self.table_name}"
        response = requests.patch(
            url,
            headers=self._headers(admin=True, prefer="return=representation"),
            params={"id": f"eq.{submission_id}"},
            json={"review_status": review_status, "admin_notes": admin_notes.strip()},
            timeout=15,
        )
        response.raise_for_status()
        rows = response.json()
        row = rows[0] if rows else {"id": submission_id, "review_status": review_status, "admin_notes": admin_notes}
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
