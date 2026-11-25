# SPDX-License-Identifier: Apache-2.0
"""One confirmed Comfy Cloud generation, with a durable job ID and no request retries."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID, uuid4

from PIL import Image

from .config import REPO_ROOT
from .guards import CircuitBreaker, PaidCallGuardError, redact_secrets
from .uv_hd_retex import digest, read_json, write_json


BASE = "https://cloud.comfy.org"
LIMIT = 500 * 1024
MAX_IMAGE_BYTES = 32 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@contextmanager
def exclusive(path):
    """Serialize request sequences across projects, releasing the OS lock on process exit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise PaidCallGuardError("A Comfy Cloud command is already active") from None
        yield


def read_key(explicit):
    filename = explicit or os.environ.get("HELLO_3DNEKO_COMFY_KEY_FILE", "").strip()
    if filename:
        try:
            key = next((line.strip() for line in Path(filename).read_text(encoding="utf-8-sig").splitlines()
                        if line.strip()), "")
        except (OSError, UnicodeError):
            raise ValueError("Comfy Cloud key file cannot be read") from None
    else:
        key = os.environ.get("COMFY_CLOUD_API_KEY", "").strip()
    if not key or not key.isascii() or any(ch.isspace() for ch in key):
        raise ValueError("A valid Comfy Cloud credential is required")
    return key


class Cloud:
    def __init__(self, key, breaker):
        self.key = key
        self.breaker = breaker
        self.http = build_opener(NoRedirect())

    def request(self, label, path, *, body=None, content_type=None, result="json"):
        self.breaker.check_before_call(label)
        signed = path.startswith("https://")
        headers = {} if signed else {"X-API-Key": self.key}
        if content_type:
            headers["Content-Type"] = content_type
        req = Request(path if signed else BASE + path, body, headers)
        accepted = False
        try:
            try:
                with self.http.open(req, timeout=45) as response:
                    cap = MAX_IMAGE_BYTES if result == "image" else LIMIT
                    payload = response.read(cap + 1)
                    if len(payload) > cap:
                        raise ValueError("response size exceeds limit")
                    if result == "image":
                        with Image.open(BytesIO(payload)) as preview:
                            if preview.format != "PNG" or max(preview.size) > 4096:
                                raise ValueError("expected PNG no larger than 4096 px")
                            preview.verify()
                        accepted = True
                        return payload
                    data = json.loads(payload)
                    if not isinstance(data, dict):
                        raise ValueError("expected JSON object")
                    accepted = True
                    return data
            except HTTPError as exc:
                with exc:
                    if result == "redirect" and exc.code == 302:
                        url = urlsplit(exc.headers.get("Location", ""))
                        host = url.hostname or ""
                        if url.scheme == "https" and (host == "storage.googleapis.com" or host.endswith(".storage.googleapis.com")):
                            accepted = True
                            return url.geturl()
                        raise PaidCallGuardError("Comfy Cloud returned an invalid storage URL") from None
                    raise PaidCallGuardError(f"Comfy Cloud {label}: HTTP {exc.code}") from None
            except (URLError, OSError, ValueError):
                raise PaidCallGuardError(redact_secrets(f"Comfy Cloud {label}: request or response failed")) from None
        finally:
            self.breaker.record_attempt(accepted)

    def upload(self, path):
        boundary = uuid4().hex
        marker = ("--" + boundary).encode("ascii")
        body = (marker + b'\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n'
                + marker + f'\r\nContent-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
                .encode("ascii") + b"Content-Type: image/png\r\n\r\n" + path.read_bytes()
                + b"\r\n" + marker + b"--\r\n")
        data = self.request("upload", "/api/upload/image", body=body,
                            content_type=f"multipart/form-data; boundary={boundary}")
        filename = data.get("name")
        if not isinstance(filename, str) or not filename:
            raise PaidCallGuardError("Comfy Cloud upload returned no image reference")
        folder = data.get("subfolder")
        return f"{folder}/{filename}" if folder else filename

    def submit(self, source, prompt):
        workflow = {
            "1": {"class_type": "LoadImage", "inputs": {"image": source}},
            "2": {"class_type": "GeminiImage2Node", "inputs": {
                "model": "gemini-3-pro-image-preview", "prompt": prompt, "seed": 42,
                "aspect_ratio": "1:1", "resolution": "2K", "response_modalities": "IMAGE",
                "images": ["1", 0],
            }},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "garment_retexture"}},
        }
        response = self.request("submit", "/api/prompt", content_type="application/json", body=json.dumps({
            "prompt": workflow, "extra_data": {"api_key_comfy_org": self.key},
        }).encode())
        try:
            return str(UUID(response.get("prompt_id") or response.get("job_id") or ""))
        except (ValueError, TypeError, AttributeError):
            raise PaidCallGuardError("Job submission outcome is ambiguous; inspect the Cloud queue") from None

    def poll_and_download(self, job_id):
        until = time.monotonic() + 480
        while time.monotonic() < until:
            job = self.request("poll", f"/api/jobs/{quote(str(UUID(job_id)), safe='')}")
            status = job.get("status")
            if status == "completed":
                outputs = job.get("outputs", {}).get("3", {}).get("images", [])
                if not outputs or not outputs[0].get("filename"):
                    raise PaidCallGuardError("The completed Comfy Cloud job returned no SaveImage result")
                image = outputs[0]
                path = "/api/view?" + urlencode({
                    "filename": image["filename"], "subfolder": image.get("subfolder", ""), "type": "output",
                })
                signed = self.request("output redirect", path, result="redirect")
                if not isinstance(signed, str):
                    raise PaidCallGuardError("Expected a signed Comfy Cloud image URL")
                return self.request("image download", signed, result="image")
            if status not in ("pending", "in_progress"):
                raise PaidCallGuardError("Comfy Cloud job failed, was cancelled, or returned an unknown status")
            time.sleep(30)
        raise PaidCallGuardError("Comfy Cloud poll timed out; the saved job can be checked with --resume")


def generate(root, args, prompt):
    state_path = root / "comfy_job_state.json"
    source = root / "inputs" / f"{args.view}.png"
    destination = root / "generated" / "comfy" / f"{args.view}.png"
    shared = REPO_ROOT / "projects" / ".comfy_retexture"
    with exclusive(shared / "requests.lock"):
        jobs = read_json(state_path) if state_path.is_file() else {}
        job = jobs.get(args.view)
        if args.resume:
            if not job or not job.get("job_id"):
                raise PaidCallGuardError("No saved job ID; inspect the Cloud queue before another submission")
            if job["status"] == "downloaded" and destination.is_file():
                return job
        else:
            cost = args.confirmed_cost_cny
            if not args.confirm_preview or cost is None or not math.isfinite(cost) or not 0 < cost <= 5:
                raise PaidCallGuardError("Confirm the input preview and a cost of at most CNY 5 before submission")
            if job:
                raise PaidCallGuardError("This view already has a submission record; use --resume")
            if not source.is_file() or not (root / "cameras.json").is_file():
                raise ValueError("Run prepare and inspect inputs/preview.png before submission")
        cloud = Cloud(read_key(args.key_file), CircuitBreaker(shared / "circuit.json"))
        if not args.resume:
            framing = ("Views in row order: front, back, left, right." if args.view == "view_grid"
                       else f"This is the {args.view} view.")
            prompt = f"{framing} {prompt}"
            prompt_id = hashlib.sha256(prompt.encode()).hexdigest()
            history_path = shared / "recent_prompts.json"
            history = read_json(history_path) if history_path.is_file() else {}
            now = time.time()
            history = {key: value for key, value in history.items() if now - value < 600}
            if prompt_id in history:
                raise PaidCallGuardError("The same Comfy Cloud prompt was submitted within ten minutes")
            remote = cloud.upload(source)
            job = {"status": "submitting", "input_sha256": digest(source), "confirmed_cost_cny": cost}
            jobs[args.view] = job
            write_json(state_path, jobs)
            history[prompt_id] = time.time()
            write_json(history_path, history)
            job.update(job_id=cloud.submit(remote, prompt), status="submitted")
            write_json(state_path, jobs)
        image_data = cloud.poll_and_download(job["job_id"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = destination.with_suffix(".tmp")
        temp.write_bytes(image_data)
        temp.replace(destination)
        job.update(status="downloaded", output=str(destination.relative_to(root)))
        write_json(state_path, jobs)
        return job
