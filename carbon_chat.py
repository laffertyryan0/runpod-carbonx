#!/usr/bin/env python3
"""Carbon-X on Runpod — one-file Linux launcher (Python 3.10+, standard library).

RUN
    python3 carbon_chat.py
    # Enter a Runpod API key at the hidden prompt, or set RUNPOD_API_KEY.
    python3 carbon_chat.py --share   # also print a guest chat link
    python3 carbon_chat.py --stop    # remove this installation's leftover Pods
    python3 carbon_chat.py --check   # read-only checks; does not rent a GPU
    python3 carbon_chat.py --version # show the launcher version
    python3 carbon_chat.py --demo    # UI preview; no cloud, no model, no charges

No local Docker, CUDA, pip packages, SSH setup, or Hugging Face token is needed.
Use a funded Runpod account and an API key with read/write access to Pods plus
read access to account information, GPU inventory, and billing. Auto-pay must
be off. This program never changes your payment settings or adds funds.

DEFAULTS
Official Carbon-X Q8_0; 8192-token context shared by one request; thinking off;
15-minute idle shutdown; 4-hour maximum session INCLUDING setup; 20-minute
startup timeout; GPU price ceiling $0.40/hour. Prices are queried before renting
and checked again immediately afterward. Storage is estimated separately.

The operating allowance is $15 per UTC calendar month, leaving $5 of the user's
$20 budget for tax, pricing/billing lag, and failures. A local reservation ledger
and Runpod's aggregated account charges are both checked.
This is NOT a provider-enforced dollar cap. Other account activity, lost local
state, concurrent launches from other computers, API failures, image startup
failures, and billing delays can defeat application-level controls. Keep auto-pay
off and limit prepaid credit. Do not delete the ledger to reset your allowance.

The Pod is DELETED at shutdown, including its temporary model download. There
are no persistent cloud volumes. The 15.8 GB model downloads each session; setup
is billable. Owner chats are saved locally in the state directory, with file mode
0600. Guest chats stay in their own browser and should be exported before the
session ends; the share URL changes between sessions. Chat history is sent to
the model every turn. Overlong conversations are rejected, never silently cut.

The Runpod key is held in local memory AND passed to the private Pod environment
for its self-deletion watchdog. It is never sent to browser JavaScript or guests,
written to the local ledger, or inherited by llama-server. Use a dedicated key.
The watchdog starts only after the container boots. Local monitoring covers
image-pull/startup failures while this launcher is running. Ctrl-C and End session
delete the Pod. A failed deletion is reported and retried; verify in Runpod if
the launcher cannot confirm cleanup. Only Pods with this installation's exact
name and ownership marker are managed.

All model text is rendered literally, including DNA tags and code. No remote
JavaScript, analytics, generated HTML execution, or model-requested actions.

Sources verified 2026-10-22:
https://huggingface.co/HuggingFaceBio/Carbon-X-GGUF
https://github.com/ggml-org/llama.cpp/blob/b11742/.devops/cuda.Dockerfile
https://api.runpod.io/v2/openapi.json
https://docs.runpod.io/api-reference-v2/migrate-from-v1
https://docs.runpod.io/sdks/graphql/manage-pods
https://docs.runpod.io/pods/pricing
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import datetime as dt
import fcntl
import getpass
import hashlib
import hmac
import http.server
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zlib

VERSION = "1.1.0"
REST = "https://api.runpod.io/v2"
GRAPHQL = "https://api.runpod.io/graphql"
MODEL_REPO = "HuggingFaceBio/Carbon-X-GGUF"
MODEL_REVISION = "cbb978c02e494bbe4043ed582c532296fc97a31e"
MODEL_FILE = "carbon-x-q8_0.gguf"
MODEL_SHA256 = "48241178b67ab5423a74f65552dd04628ac16f2fa5d1a60dc94019d86fa6f453"
MODEL_BYTES = 15755742080
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}"
MIN_LLAMA_BUILD = 11742
# Official linux/amd64 full-CUDA image, llama.cpp b11804, CUDA 12.8.1.
# Keep deployments reproducible instead of pulling a changing latest tag.
IMAGE_DIGEST = "sha256:0f5ab9182a7fce171f73226373627424303886f23b5457e2353ed139b098cc9d"
ALLOWANCE = 15.0
CONTEXT = 8192
MAX_OUTPUT = 2048
DISK_GB = 50
DISK_RATE = DISK_GB * 0.10 / (30 * 24)
MAX_BODY = 2 * 1024 * 1024
MIN_CUDA_VERSION = "12.8"
USER_AGENT = "carbon-x-runpod-launcher/" + VERSION


class AppError(Exception):
    pass


class HttpError(AppError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # Never forward a Runpod/chat credential to a redirect target.


PRIVATE_OPENER = urllib.request.build_opener(NoRedirect())


def fetch(url, *, method="GET", data=None, token=None, timeout=30):
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        data = json.dumps(data, allow_nan=False).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        return PRIVATE_OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raw = exc.read(4096).decode("utf-8", "replace")
        if token:
            raw = raw.replace(token, "[redacted]")
        # Provider validation errors can echo complete env/body. Do not print them.
        if "CARBON_" in raw or "env" in raw or "api_key" in raw:
            raw = "Provider rejected the request; check key permissions and account balance."
        try:
            obj = json.loads(raw)
            raw = str(obj.get("detail", obj.get("error", obj.get("message", obj.get("title", "Request rejected")))))
        except (ValueError, AttributeError):
            raw = "Request rejected" if "<" in raw else raw
        raise HttpError(exc.code, f"HTTP {exc.code}: {raw[:240]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AppError(f"Connection failed to {urllib.parse.urlsplit(url).hostname}: {type(exc).__name__}") from None


def get_json(url, **kwargs):
    with fetch(url, **kwargs) as response:
        raw = response.read(8 * 1024 * 1024)
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        raise AppError("Service returned invalid JSON.") from None


def finite_number(value, label, *, positive=False):
    if isinstance(value, bool):
        raise AppError(f"Invalid {label}.")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise AppError(f"Missing or invalid {label}.") from None
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise AppError(f"Invalid {label}.")
    return result


def atomic_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + "." + secrets.token_hex(4) + ".tmp")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(obj, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default
    except (ValueError, OSError):
        raise AppError(f"Cannot read {path}. Preserve this file; do not reset the spending ledger.") from None


def month_bounds(now=None):
    now = time.time() if now is None else now
    stamp = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    first = stamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    following = (first.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return first.timestamp(), following.timestamp()


def iso_time(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).isoformat().replace("+00:00", "Z")


class Ledger:
    """Reserve an entire session before creation; reconcile only confirmed cleanup."""
    def __init__(self, directory):
        self.path = Path(directory) / "ledger.json"
        self.data = read_json(self.path)
        if self.data is None:
            self.data = {"version": 1, "installation": secrets.token_hex(8), "sessions": []}
            self.save()
        if (self.data.get("version") != 1 or
                not re.fullmatch(r"[a-f0-9]{16}", self.data.get("installation", "")) or
                not isinstance(self.data.get("sessions"), list)):
            raise AppError("Invalid spending ledger; refusing to create a Pod.")

    @property
    def name(self):
        return "carbon-x-" + self.data["installation"]

    def save(self):
        atomic_json(self.path, self.data)

    def estimates(self, now=None, live_entry=None):
        start, end = month_bounds(now)
        totals = {}
        for item in self.data["sessions"]:
            a = finite_number(item.get("start"), "session start")
            b = finite_number(item.get("end", item.get("deadline")), "session end")
            if item is live_entry:
                b = min(b, time.time() if now is None else now)
            rate = finite_number(item.get("rate"), "session price", positive=True)
            key = item.get("pod_id") or "pending:" + item["id"]
            totals[key] = totals.get(key, 0.0) + max(0.0, min(b, end) - max(a, start)) * rate / 3600
        return totals

    def used(self, now=None):
        return sum(self.estimates(now).values())

    def reserve(self, start, deadline, rate):
        entry = {"id": secrets.token_hex(8), "start": start, "deadline": deadline,
                 "rate": rate, "pod_id": None, "closed": False}
        self.data["sessions"].append(entry)
        self.save()
        return entry

    def close(self, entry, now=None):
        # Also account for a small shutdown/billing buffer.
        entry["end"] = (time.time() if now is None else now) + 120
        entry["closed"] = True
        self.save()


class Runpod:
    def __init__(self, key):
        self.key = key
        self.candidates = []

    def rest(self, path, method="GET", data=None):
        return get_json(REST + path, method=method, data=data, token=self.key)

    def graphql(self, query):
        result = get_json(GRAPHQL, method="POST", data={"query": query}, token=self.key)
        if not isinstance(result, dict) or result.get("errors") or not result.get("data"):
            raise AppError("Runpod GraphQL query failed. The key needs account and GPU inventory read access.")
        return result["data"]

    def account(self):
        # REST v2 does not expose credit balance or auto-pay. This read-only
        # account query is the sole remaining GraphQL dependency.
        obj = self.graphql("query { myself { clientBalance isAutoPayEnabled } }").get("myself")
        if not isinstance(obj, dict) or not isinstance(obj.get("isAutoPayEnabled"), bool):
            raise AppError("Could not verify account auto-pay setting; no Pod created.")
        if obj["isAutoPayEnabled"]:
            raise AppError("Auto-pay is enabled. Turn it off once at https://console.runpod.io/billing, then rerun. This launcher will not change payment settings.")
        return finite_number(obj.get("clientBalance"), "Runpod credit balance")

    def billing(self, now=None):
        now = time.time() if now is None else now
        start, _ = month_bounds(now)
        params = urllib.parse.urlencode({"startTime": iso_time(start), "endTime": iso_time(now), "bucketSize": "month"})
        total, pods = 0.0, {}
        # v2 /billing includes all resource types. The separate Pod breakdown
        # reconciles our local reservations without hiding other account spend.
        for route in ("/billing", "/billing/pods"):
            result = self.rest(route + "?" + params)
            if not isinstance(result, dict) or not isinstance(result.get("records"), list):
                raise AppError("Unexpected billing response; cannot verify monthly spending.")
            records = result["records"]
            metadata = result.get("metadata", {})
            if not isinstance(metadata, dict) or metadata.get("recordCount") != len(records):
                raise AppError("Incomplete billing response; refusing to guess monthly spending.")
            subtotal = 0.0
            for row in records:
                if not isinstance(row, dict):
                    raise AppError("Unexpected billing row; refusing to guess spending.")
                amount = finite_number(row.get("totalAmount"), "billing amount")
                subtotal += amount
                if route == "/billing/pods":
                    pod_id = row.get("podId")
                    if not isinstance(pod_id, str) or not pod_id:
                        raise AppError("Pod billing response has no Pod ID.")
                    pods[pod_id] = pods.get(pod_id, 0.0) + amount
            declared = finite_number(metadata.get("totals", {}).get("totalAmount"), "billing total")
            if not math.isclose(declared, subtotal, abs_tol=0.001):
                raise AppError("Billing records and total disagree; cannot verify monthly spending.")
            total = max(total, declared, subtotal)
        return {"total": total, "pods": pods}

    def choices(self, ceiling):
        self.candidates = []
        choices = []
        for cloud in ("SECURE", "COMMUNITY"):
            params = urllib.parse.urlencode({"include": "AVAILABILITY", "product": "POD",
                "count": 1, "cloud": cloud, "minCudaVersion": MIN_CUDA_VERSION})
            result = self.rest("/catalog/gpus?" + params)
            if not isinstance(result, dict) or not isinstance(result.get("gpus"), list):
                raise AppError("Could not read the Runpod GPU catalog.")
            for gpu in result["gpus"]:
                if not isinstance(gpu, dict) or gpu.get("manufacturer") != "NVIDIA":
                    continue
                try:
                    memory = finite_number(gpu.get("memory"), "GPU VRAM")
                    rate = finite_number(gpu.get("price", {}).get(cloud.lower()), "GPU price", positive=True)
                except AppError:
                    continue
                if memory < 24 or not isinstance(gpu.get("id"), str):
                    continue
                stock = gpu.get("availability", "UNKNOWN")
                offered = gpu.get(cloud.lower()) is True
                available = offered and stock in ("LOW", "MEDIUM", "HIGH")
                reason = "eligible" if available and rate <= ceiling else (
                    "not offered in this cloud" if not offered else
                    "capacity not reported" if stock == "UNKNOWN" else
                    "no compatible capacity" if not available else "above hourly price ceiling")
                candidate = {"gpu": gpu["id"], "name": gpu.get("name", gpu["id"]), "cloud": cloud,
                    "rate": rate, "memory": int(memory), "available": available, "stock": stock, "reason": reason}
                self.candidates.append(candidate)
                if reason == "eligible":
                    choices.append(candidate)
        return sorted(choices, key=lambda c: (c["rate"], c["cloud"] != "SECURE"))

    def describe_candidates(self, ceiling):
        print(f"Catalog check: NVIDIA 24 GB+, CUDA {MIN_CUDA_VERSION}+, one GPU; ceiling ${ceiling:.2f}/hour.", flush=True)
        if not self.candidates:
            print("Runpod returned no priced NVIDIA GPU with at least 24 GB VRAM.", flush=True)
            return
        rows = sorted(self.candidates, key=lambda r: (not r["available"], r["rate"]))[:10]
        for row in rows:
            print(f"  {row['name']} / {row['cloud'].lower()}: ${row['rate']:.3f}/hour, "
                  f"{row['memory']} GB, capacity {row['stock']}: {row['reason']}", flush=True)
        available = [r for r in self.candidates if r["available"]]
        if available:
            cheapest = min(available, key=lambda r: r["rate"])
            print(f"Cheapest reported available option: ${cheapest['rate']:.3f}/hour. "
                  "A higher hourly ceiling buys fewer hours within the same monthly allowance.", flush=True)

    def owned(self, name, marker):
        matches, cursor, seen = [], None, set()
        while True:
            query = {"limit": 1000}
            if cursor:
                query["cursor"] = cursor
            result = self.rest("/pods?" + urllib.parse.urlencode(query))
            if not isinstance(result, dict) or not isinstance(result.get("pods"), list):
                raise AppError("Unexpected Pod inventory response.")
            matches.extend(p for p in result["pods"] if isinstance(p, dict) and p.get("name") == name and
                isinstance(p.get("env"), dict) and p["env"].get("CARBON_INSTALLATION") == marker)
            page = result.get("pagination", {})
            if not isinstance(page, dict) or not isinstance(page.get("hasNextPage"), bool):
                raise AppError("Incomplete Pod inventory pagination; cleanup cannot be verified.")
            if not page["hasNextPage"]:
                return matches
            cursor = page.get("nextCursor")
            if not isinstance(cursor, str) or not cursor or cursor in seen:
                raise AppError("Invalid Pod inventory cursor; cleanup cannot be verified.")
            seen.add(cursor)

    def delete(self, pod_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", pod_id):
            raise AppError("Invalid Pod identifier.")
        try:
            self.rest("/pods/" + pod_id, method="DELETE")
        except HttpError as exc:
            if exc.code != 404:
                raise
        # Confirm the object is gone/terminated, not merely that DELETE was accepted.
        for _ in range(8):
            try:
                pod = self.rest("/pods/" + pod_id)
                if isinstance(pod, dict) and pod.get("status") == "TERMINATED":
                    return
            except HttpError as exc:
                if exc.code == 404:
                    return
                raise
            time.sleep(1)
        raise AppError("Runpod has not yet confirmed termination.")


def verify_model():
    meta = get_json(f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}?blobs=true")
    rows = [v for v in meta.get("siblings", []) if v.get("rfilename") == MODEL_FILE]
    if (meta.get("sha") != MODEL_REVISION or meta.get("private") or meta.get("gated") or
            len(rows) != 1 or rows[0].get("lfs", {}).get("sha256") != MODEL_SHA256 or
            rows[0].get("size") != MODEL_BYTES):
        raise AppError("The pinned public Carbon-X model could not be verified; no Pod created.")


def resolve_image():
    """Verify the pinned official CUDA image before GPU billing."""
    scope = urllib.parse.urlencode({"service": "ghcr.io", "scope": "repository:ggml-org/llama.cpp:pull"})
    auth = get_json("https://ghcr.io/token?" + scope).get("token")
    if not auth:
        raise AppError("Could not read the public llama.cpp image registry.")
    accept = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json"

    def registry(path):
        req = urllib.request.Request("https://ghcr.io/v2/ggml-org/llama.cpp/" + path,
            headers={"Authorization": "Bearer " + auth, "Accept": accept, "User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=30) as response:
            raw = response.read(4 * 1024 * 1024)
            return json.loads(raw), "sha256:" + hashlib.sha256(raw).hexdigest()

    try:
        manifest, digest = registry("manifests/" + IMAGE_DIGEST)
        config, config_digest = registry("blobs/" + manifest["config"]["digest"])
        version = str(config.get("config", {}).get("Labels", {}).get("org.opencontainers.image.version", ""))
        match = re.search(r"(?:^|\D)(\d{4,})(?:\D|$)", version)
        if (digest != IMAGE_DIGEST or config_digest != manifest["config"]["digest"] or
                config.get("architecture") != "amd64" or config.get("os") != "linux" or
                "CUDA_VERSION=12.8.1" not in config.get("config", {}).get("Env", []) or
                not match or int(match[1]) < MIN_LLAMA_BUILD):
            raise AppError("Pinned CUDA image verification failed; no Pod created.")
        return "ghcr.io/ggml-org/llama.cpp@" + digest
    except urllib.error.HTTPError as exc:
        raise AppError(f"Image registry returned HTTP {exc.code}; no Pod created.") from None
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        raise AppError("Could not verify the pinned official CUDA image; no Pod created.") from None


def actual_rate(pod):
    if not isinstance(pod, dict) or pod.get("cost") is None:
        raise AppError("Runpod did not return an hourly price; cleaning up the Pod.")
    # v2 reports the effective hourly rate as cost. Reserve disk separately too.
    return finite_number(pod["cost"], "Pod hourly price", positive=True) + DISK_RATE


def reconciled_spend(report, ledger, now=None, live_entry=None):
    """Keep unrelated account charges AND reserve unreported cost of our Pods."""
    return report["total"] + sum(max(0.0, amount - report["pods"].get(pod, 0.0))
        for pod, amount in ledger.estimates(now, live_entry).items())


def validate_messages(data):
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
        raise AppError("Send between 1 and 256 chat messages.")
    cleaned, chars = [], 0
    for index, item in enumerate(messages):
        role = "user" if index % 2 == 0 else "assistant"
        if not isinstance(item, dict) or item.get("role") != role or not isinstance(item.get("content"), str):
            raise AppError("Chat messages must alternate user and assistant.")
        if not item["content"].strip():
            raise AppError("Empty chat messages are not supported.")
        chars += len(item["content"])
        cleaned.append({"role": role, "content": item["content"]})
    if cleaned[-1]["role"] != "user" or chars > 300000:
        raise AppError("Invalid or oversized conversation. Start a new chat or shorten the input.")
    return cleaned


UI = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Carbon-X · Chat</title>
<style>
:root{color-scheme:dark;--bg:#101916;--panel:#17241e;--line:#304338;--text:#e8efe9;--muted:#a4b5a9;--accent:#b3e6aa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.6 system-ui,sans-serif}button,textarea,input{font:inherit}button{cursor:pointer;border:1px solid var(--line);border-radius:10px;padding:9px 15px;color:var(--text);background:var(--panel)}button:hover{border-color:var(--accent)}button:disabled{opacity:.45;cursor:default}.primary{background:var(--accent);color:#152719;border:0;font-weight:650}.danger{color:#ffbeb1}a{color:var(--accent)}header{border-bottom:1px solid var(--line);padding:18px 28px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{font-size:21px;font-weight:650;letter-spacing:-.5px}.brand span{color:var(--accent)}.sub{color:var(--muted);font-size:12px;letter-spacing:.8px;text-transform:uppercase}.actions{display:flex;gap:8px;flex-wrap:wrap}main{max-width:900px;margin:auto;padding:24px 24px 180px}.status{display:flex;justify-content:space-between;gap:14px;color:var(--muted);font-size:13px;margin-bottom:22px}.dot{color:var(--accent)}#cost{text-align:right}.notice{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 16px;color:var(--muted);margin-bottom:20px;white-space:pre-wrap}.intro{padding:65px 0 35px}.intro h1{font-size:38px;line-height:1.2;font-weight:550;letter-spacing:-1.5px;max-width:650px;margin:0 0 18px}.intro p{color:var(--muted);max-width:600px}.suggestions{display:flex;gap:10px;flex-wrap:wrap;margin-top:25px}.suggestions button{font-size:13px}.message{margin:22px 0;padding:18px 20px;border:1px solid var(--line);border-radius:14px;white-space:pre-wrap;overflow-wrap:anywhere}.message.user{background:#1b2d23;margin-left:35px}.message.assistant{background:#132019;margin-right:35px}.who{font-size:11px;font-weight:700;letter-spacing:1px;color:var(--accent);display:block;margin-bottom:10px}.composer{position:fixed;bottom:0;left:0;right:0;background:linear-gradient(transparent,var(--bg) 18%);padding:28px 24px 18px}.compose-inner{max-width:852px;margin:auto;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:14px}.compose-bottom{display:flex;align-items:center;justify-content:space-between;gap:10px}textarea{display:block;width:100%;resize:vertical;min-height:65px;max-height:230px;border:0;background:transparent;color:var(--text);outline:none}small{font-size:11px;color:var(--muted)}#error{color:#ffbeb1;white-space:pre-wrap;margin:8px 0}#login{max-width:520px;margin:50px auto}input{width:100%;border:1px solid var(--line);padding:12px;background:var(--panel);color:var(--text);border-radius:8px;margin:12px 0}.hidden{display:none!important}#shareBox{overflow-wrap:anywhere}select{background:var(--panel);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:6px;max-width:260px}.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:center}.note{color:var(--muted);font-size:12px;margin-top:10px} @media(max-width:650px){header{padding:14px 16px;align-items:flex-start}.brand{font-size:18px}.actions button{padding:6px 9px;font-size:12px}main{padding:18px 16px 190px}.intro{padding-top:28px}.intro h1{font-size:29px}.message.user{margin-left:10px}.message.assistant{margin-right:10px}.status{flex-direction:column}#cost{text-align:left}.composer{padding:28px 12px 12px}}
</style>
<header><div><div class="brand">Carbon<span>-X</span></div><div class="sub">DNA & language</div></div><div class="actions"><button id="new">New chat</button><button id="share" class="hidden">Share session</button><button id="end" class="danger hidden">End session</button></div></header>
<main><div id="login" class="hidden"><h2>Open your session</h2><p>Enter the access code from your launcher or use the full invitation link.</p><input id="access" type="password" placeholder="Session access code" autocomplete="off"><button id="signin" class="primary">Open chat</button></div>
<div id="app"><div class="status"><span id="state">Connecting…</span><span id="cost"></span></div><div id="notice" class="notice hidden"></div><div id="shareBox" class="notice hidden"></div><div class="toolbar"><select id="chats" aria-label="Saved chats"></select><button id="export">Export</button><button id="import">Import</button><button id="retry" class="hidden">Retry reply</button><input id="file" type="file" accept="application/json,.json" class="hidden"></div><div id="messages"></div><div id="error" role="alert"></div></div></main>
<div id="composer" class="composer"><div class="compose-inner"><textarea id="prompt" aria-label="Message" placeholder="Ask a question, or paste a DNA sequence…"></textarea><div class="compose-bottom"><small>Enter to send · Shift+Enter for a new line</small><div><button id="cancel" class="hidden">Stop reply</button><button id="send" class="primary" disabled>Send</button></div></div><div class="note" id="historyNote">Your conversation is saved on this device.</div></div></div>
<script>
'use strict';
const $=id=>document.getElementById(id);let token=new URLSearchParams(location.hash.slice(1)).get('access')||sessionStorage.getItem('carbon_access')||'';
if(token){sessionStorage.setItem('carbon_access',token);history.replaceState(null,'',location.pathname)}
let saved=[],current=null,busy=false,abort=null,ready=false,role='guest',local=false,saveTimer=null,lastActivity=0,offline=false,saveRunning=false,saveDirty=false;
const uid=()=>crypto.randomUUID();const currentChat=()=>saved.find(c=>c.id===current);
function error(t){$('error').textContent=t||''}
async function api(path,options={}){const r=await fetch(path,{...options,headers:{'Content-Type':'application/json','Authorization':'Bearer '+token,...options.headers}});if(!r.ok){let d;try{d=await r.json()}catch{d={error:'Connection error ('+r.status+')'}}if(r.status===401)login();throw Error(d.error||'Request failed')}return r}
function login(){$('login').classList.remove('hidden');$('app').classList.add('hidden');$('composer').classList.add('hidden')}
function showApp(){$('login').classList.add('hidden');$('app').classList.remove('hidden');$('composer').classList.remove('hidden')}
async function flushSave(){if(saveRunning||!local)return;saveRunning=true;try{while(saveDirty){saveDirty=false;await api('/api/history',{method:'POST',body:JSON.stringify({chats:saved})})}}catch(e){saveDirty=true;error('Local save failed: '+e.message)}finally{saveRunning=false}}
function persist(){try{localStorage.setItem('carbon_chats',JSON.stringify(saved))}catch{error('Browser storage is full. Export your conversation now.')}if(local){saveDirty=true;clearTimeout(saveTimer);saveTimer=setTimeout(flushSave,150)}}
function render(){const c=currentChat();const area=$('messages');area.replaceChildren();$('chats').replaceChildren();for(const chat of saved){const o=document.createElement('option');o.value=chat.id;o.textContent=chat.title||'New conversation';o.selected=chat.id===current;$('chats').append(o)}if(!c||!c.messages.length){const intro=document.createElement('div');intro.className='intro';intro.innerHTML='<div class="sub">Your sequence. Your questions.</div><h1>A conversation about DNA.</h1><p>Ask about a sequence, explore an idea, or follow up on a previous answer. Include any gene, organism, or other context you already know.</p><div class="suggestions"><button>Explain the difference between DNA and RNA.</button><button>What context should I provide with a DNA sequence?</button></div>';for(const b of intro.querySelectorAll('button'))b.onclick=()=>{$('prompt').value=b.textContent;$('prompt').focus()};area.append(intro)}else for(const m of c.messages){const box=document.createElement('div');box.className='message '+m.role;const who=document.createElement('span');who.className='who';who.textContent=m.role==='user'?'YOU':'CARBON-X';box.append(who,document.createTextNode(m.content||'…'));area.append(box)}$('send').disabled=!ready||busy;$('new').disabled=busy;$('chats').disabled=busy;$('import').disabled=busy;$('cancel').classList.toggle('hidden',!busy);$('retry').classList.toggle('hidden',busy||!c||c.messages.at(-1)?.role!=='user');$('retry').disabled=!ready}
function newChat(){if(busy)return;if(saved.length>=100){error('Export your chats first: the 100-chat limit has been reached.');return}const c={id:uid(),title:'New conversation',messages:[],updated:Date.now()};saved.unshift(c);current=c.id;persist();render();$('prompt').focus()}
function validChats(x){if(!Array.isArray(x)||x.length>100)return false;return x.every(c=>c&&typeof c.id==='string'&&typeof c.title==='string'&&Array.isArray(c.messages)&&c.messages.length<=256&&c.messages.every(m=>m&&['user','assistant'].includes(m.role)&&typeof m.content==='string'))}
async function init(){if(!token){login();return}showApp();try{const s=await (await api('/api/status')).json();local=s.local===true;role=s.role;let data;if(local)data=(await (await api('/api/history')).json()).chats;else{try{data=JSON.parse(localStorage.getItem('carbon_chats')||'[]')}catch{data=[]}}saved=validChats(data)?data:[];if(!saved.length)newChat();else{current=saved[0].id;render()}updateStatus(s)}catch(e){error(e.message)}}
function updateStatus(s){offline=s.phase==='stopped'||s.phase==='stopping';ready=s.phase==='ready';$('state').textContent=(ready?'● Ready · ':s.phase==='demo'?'Preview · ':'● ')+(s.message||s.phase);if(s.phase==='demo')ready=true;$('cost').textContent=s.hourly_rate?'$'+s.hourly_rate.toFixed(3)+'/hour estimated · '+Math.max(0,Math.ceil(s.remaining_seconds/60))+' min left':'';$('notice').classList.toggle('hidden',ready&&!s.notice);$('notice').textContent=s.notice||s.message||'';$('end').classList.toggle('hidden',s.role!=='owner'||offline);$('share').classList.toggle('hidden',!s.share_url||offline);$('share').onclick=()=>{$('shareBox').classList.toggle('hidden');$('shareBox').textContent='Guest link (valid while this session runs):\n'+s.share_url+'\nGuests have their own chat history and cannot end your session.'};$('historyNote').textContent=s.local?'Chats saved on your Linux machine. The full history is sent with each turn.':'Chats saved in this browser. Export before this session ends to use them at a new link.';$('send').disabled=!ready||busy;$('retry').disabled=!ready||busy}
async function poll(){if(!token)return;try{updateStatus(await (await api('/api/status')).json())}catch(e){ready=false;$('send').disabled=true;$('state').textContent='Disconnected — your chat remains saved';if(!(e.message||'').includes('Unauthorized'))error(e.message)}}
async function send(){const text=$('prompt').value.trim();if(!text||busy||!ready)return;error('');let c=currentChat();if(c.messages.length&&c.messages.at(-1).role==='user'){error('The previous reply failed. Use Retry below, or start a new chat.');return}c.messages.push({role:'user',content:text});if(c.messages.length===1)c.title=text.slice(0,55);c.updated=Date.now();$('prompt').value='';persist();await generate(c)}
async function generate(c){busy=true;abort=new AbortController();const requestMessages=c.messages.map(m=>({...m}));const reply={role:'assistant',content:''};c.messages.push(reply);render();let finished=false;try{const r=await api('/api/chat',{method:'POST',body:JSON.stringify({messages:requestMessages}),signal:abort.signal});const reader=r.body.getReader(),decoder=new TextDecoder();let buffer='';while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});buffer=buffer.replace(/\r\n/g,'\n');let ix;while((ix=buffer.indexOf('\n\n'))>=0){const frame=buffer.slice(0,ix);buffer=buffer.slice(ix+2);for(const line of frame.split('\n')){if(!line.startsWith('data:'))continue;const raw=line.slice(5).trim();if(raw==='[DONE]'){finished=true;continue}const d=JSON.parse(raw);if(d.error)throw Error(typeof d.error==='string'?d.error:d.error.message||'Generation failed');const ch=d.choices?.[0];reply.content+=ch?.delta?.content||'';if(ch?.finish_reason==='length')error('Reply reached its length limit. You can ask it to continue.')}}render();persist();window.scrollTo({top:document.body.scrollHeight,behavior:'auto'})}if(!finished)throw Error('The connection ended before the reply completed. Your received text is saved.')}catch(e){if(e.name!=='AbortError')error(e.message);else error('Reply stopped. Received text has been saved.')}finally{if(!reply.content){c.messages.pop();}busy=false;abort=null;persist();render()}}
$('retry').onclick=()=>{error('');generate(currentChat())};$('send').onclick=send;$('cancel').onclick=()=>abort?.abort();$('prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send()}if(Date.now()-lastActivity>20000){lastActivity=Date.now();api('/api/activity',{method:'POST',body:'{}'}).catch(()=>{})}};$('new').onclick=newChat;$('chats').onchange=()=>{current=$('chats').value;render()};
$('end').onclick=async()=>{if(!confirm('End this GPU session? Your saved conversations will remain.'))return;abort?.abort();try{await api('/api/end',{method:'POST',body:'{}'});ready=false;$('send').disabled=true;$('state').textContent='Ending session…'}catch(e){error(e.message)}};
$('export').onclick=()=>{const blob=new Blob([JSON.stringify({chats:saved},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='carbon-x-chats.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
$('import').onclick=()=>$('file').click();$('file').onchange=async()=>{try{const f=$('file').files[0];if(!f||f.size>15000000)throw Error('Select a chat export smaller than 15 MB.');const data=JSON.parse(await f.text());if(!validChats(data.chats))throw Error('Invalid chat export.');if(saved.length+data.chats.length>100)throw Error('At most 100 saved chats are supported.');for(const c of data.chats)saved.unshift({...c,id:uid()});current=saved[0].id;persist();render()}catch(e){error(e.message)}$('file').value=''};
$('signin').onclick=()=>{token=$('access').value.trim();sessionStorage.setItem('carbon_access',token);init()};init();setInterval(poll,5000);
</script></html>'''


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "CarbonChat"

    def log_message(self, *args):
        pass  # No prompts, credential-bearing URLs, or headers in access logs.

    @property
    def app(self):
        return self.server.app

    def headers_common(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

    def respond(self, code, value, content_type="application/json"):
        raw = value.encode() if isinstance(value, str) else json.dumps(value, allow_nan=False).encode()
        self.send_response(code)
        self.headers_common()
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def role(self):
        header = self.headers.get("Authorization", "")
        if header.startswith("Bearer "):
            token = header[7:]
            if hmac.compare_digest(token.encode(), self.app.owner.encode()):
                return "owner"
            if self.app.guest and hmac.compare_digest(token.encode(), self.app.guest.encode()):
                return "guest"
        return None

    def body(self, limit=MAX_BODY):
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            raise AppError("Invalid request size.") from None
        if self.headers.get("Transfer-Encoding") or not 0 <= length <= limit:
            raise AppError("Request is too large or missing its size.")
        self.connection.settimeout(15)
        try:
            return json.loads(self.rfile.read(length))
        except (ValueError, OSError):
            raise AppError("Invalid JSON request.") from None

    def do_GET(self):
        try:
            if not self.host_allowed():
                return self.respond(403, {"error": "Invalid host."})
            if self.path == "/":
                return self.respond(200, UI, "text/html")
            role = self.role()
            if not role:
                return self.respond(401, {"error": "Unauthorized session access."})
            if self.path == "/api/status":
                return self.respond(200, self.app.status(role))
            if self.path == "/api/history" and self.app.local and role == "owner":
                return self.respond(200, {"chats": self.app.history()})
            return self.respond(404, {"error": "Not found."})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except AppError as exc:
            self.respond(503, {"error": str(exc)})

    def host_allowed(self):
        if not self.app.local:
            return True
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

    def do_POST(self):
        try:
            if not self.host_allowed():
                return self.respond(403, {"error": "Invalid host."})
            role = self.role()
            if not role:
                self.close_connection = True
                return self.respond(401, {"error": "Unauthorized session access."})
            data = self.body(16 * 1024 * 1024 if self.path == "/api/history" and self.app.local else MAX_BODY)
            if self.path == "/api/history" and self.app.local and role == "owner":
                self.app.save_history(data)
                return self.respond(200, {"ok": True})
            if self.path == "/api/end":
                if role != "owner":
                    return self.respond(403, {"error": "Only the session owner can end it."})
                self.respond(200, {"ok": True})
                threading.Thread(target=self.app.end, args=("Session ended by owner",), daemon=True).start()
                return
            if self.path == "/api/activity":
                self.app.activity(role)
                return self.respond(200, {"ok": True})
            if self.path == "/api/chat":
                messages = validate_messages(data)
                return self.app.chat(self, messages, role)
            return self.respond(404, {"error": "Not found."})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except AppError as exc:
            self.close_connection = True
            self.respond(400, {"error": str(exc)})

    def sse_start(self):
        self.send_response(200)
        self.headers_common()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.connection.settimeout(30)
        self.wfile.write(b": connected\n\n")
        self.wfile.flush()

    def sse(self, value):
        raw = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
        self.wfile.write(("data: " + raw + "\n\n").encode())
        self.wfile.flush()


def pipe_sse(handler, connect, timeout=600):
    """Keep the proxy connection alive during prefill; close upstream on abort."""
    events = queue.Queue(maxsize=128)
    canceled = threading.Event()
    holder = []

    def put(item):
        while not canceled.is_set():
            try:
                events.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    def read():
        try:
            with connect() as response:
                holder.append(response)
                while not canceled.is_set():
                    line = response.readline(2 * 1024 * 1024)
                    if not line:
                        break
                    put(line)
        except Exception as exc:
            put({"error": str(exc) if isinstance(exc, AppError) else "Model connection interrupted."})
        finally:
            put(None)

    handler.sse_start()
    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline, last_ping = time.monotonic() + timeout, time.monotonic()
    try:
        while time.monotonic() < deadline:
            try:
                event = events.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() - last_ping >= 5:
                    handler.wfile.write(b": waiting\n\n")
                    handler.wfile.flush()
                    last_ping = time.monotonic()
                continue
            if event is None:
                return
            if isinstance(event, dict):
                handler.sse(event)
                return
            handler.wfile.write(event)
            handler.wfile.flush()
        handler.sse({"error": "Reply exceeded the 10-minute limit. Received text was saved."})
    except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
        pass
    finally:
        canceled.set()
        for response in holder:
            try:
                # Interrupt a blocked upstream read promptly; close() alone can wait.
                response.fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError, OSError):
                pass
            try:
                response.close()
            except OSError:
                pass


class Worker:
    local = False
    def __init__(self, settings, api, pod_id):
        self.settings, self.api, self.pod_id = settings, api, pod_id
        self.owner, self.guest = settings["owner"], settings.get("guest", "")
        self.phase, self.message = "starting", "Preparing model download"
        self.last_activity = time.monotonic()
        self.ready_at = None
        self.active = 0
        self.generation = threading.Lock()
        self.stop_lock = threading.Lock()
        self.process = None
        self.requests = collections.deque()
        self.llama = "http://127.0.0.1:8081"
        self.model_dir = Path("/tmp")

    def status(self, role):
        return {"phase": self.phase, "message": self.message, "role": role, "local": False,
                "remaining_seconds": max(0, self.settings["deadline"] - time.time()),
                "hourly_rate": self.settings["rate"], "share_url": None}

    def activity(self, role):
        self.last_activity = time.monotonic()

    def end(self, reason):
        if not self.stop_lock.acquire(blocking=False):
            return
        self.phase, self.message = "stopping", reason
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except OSError:
                pass  # A model-exit race must never prevent Pod deletion.
        print("Stopping Carbon-X Pod: " + reason, flush=True)
        while True:
            try:
                self.api.delete(self.pod_id)
                self.phase = "stopped"
                return
            except AppError:
                print("Termination not confirmed; retrying. Check Runpod console if this persists.", flush=True)
                time.sleep(10)

    def watchdog(self):
        while self.phase not in ("stopping", "stopped"):
            now = time.time()
            if now >= self.settings["deadline"]:
                self.end("Maximum session duration reached")
                return
            if self.ready_at is None and now >= self.settings["startup_deadline"]:
                self.end("Model startup timed out")
                return
            if self.ready_at is not None and not self.active and time.monotonic() - self.last_activity >= self.settings["idle_seconds"]:
                self.end("No chat activity for 15 minutes" if self.settings["idle_seconds"] == 900 else "Idle timeout reached")
                return
            if self.phase == "ready" and self.process and self.process.poll() is not None:
                self.end("Model server stopped unexpectedly")
                return
            time.sleep(2)

    def boot(self):
        try:
            # Validate credentials/ownership early; the key is not inherited by llama.
            pod = self.api.rest("/pods/" + self.pod_id)
            if pod.get("env", {}).get("CARBON_INSTALLATION") != self.settings["installation"]:
                raise AppError("Pod ownership verification failed")
            rate = actual_rate(pod)
            if rate > self.settings["max_rate"]:
                raise AppError("Assigned Pod exceeds the price ceiling")
            check = subprocess.run(["/app/llama-server", "--version"], capture_output=True, text=True, timeout=30)
            text = check.stdout + check.stderr
            found = re.search(r"(?:version:|build:)\s*b?(\d+)", text)
            if check.returncode or not found or int(found[1]) < MIN_LLAMA_BUILD:
                raise AppError("llama.cpp build is too old or could not be verified")
            model = self.model_dir / MODEL_FILE
            digest, received, last_progress = hashlib.sha256(), 0, 0.0
            self.message = "Downloading Carbon-X (15.8 GB); setup time is billable"
            req = urllib.request.Request(MODEL_URL, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as response, model.open("wb") as output:
                while True:
                    if self.phase in ("stopping", "stopped"):
                        return
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    received += len(chunk)
                    if received > MODEL_BYTES:
                        raise AppError("Unexpected model download size")
                    output.write(chunk)
                    digest.update(chunk)
                    if time.monotonic() - last_progress > 2:
                        self.message = f"Downloading Carbon-X: {100 * received / MODEL_BYTES:.0f}%"
                        last_progress = time.monotonic()
            if received != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256:
                raise AppError("Model checksum verification failed")
            if self.phase in ("stopping", "stopped"):
                return
            self.message = "Loading model into GPU memory"
            args = ["/app/llama-server", "-m", str(model), "--host", "127.0.0.1", "--port", "8081",
                    "-ngl", "99", "-c", str(CONTEXT), "-np", "1", "--jinja", "--no-webui",
                    "--no-context-shift", "--flash-attn", "on", "--batch-size", "512", "--ubatch-size", "128",
                    "--chat-template-kwargs", '{"enable_thinking":false}',
                    "--temp", "0.6", "--top-p", "0.95", "--top-k", "20", "--min-p", "0", "--alias", "carbon-x"]
            child_env = {k: v for k, v in os.environ.items() if not k.startswith("CARBON_") and k != "RUNPOD_API_KEY"}
            with (self.model_dir / "carbon-model.log").open("ab", buffering=0) as log:
                self.process = subprocess.Popen(args, env=child_env, stdout=subprocess.DEVNULL, stderr=log)
            while self.phase not in ("stopping", "stopped"):
                if self.process.poll() is not None:
                    raise AppError("Model server failed to load; inspect /tmp/carbon-model.log in Pod logs before retrying")
                healthy = False
                try:
                    healthy = get_json(self.llama + "/health", timeout=3).get("status") == "ok"
                except HttpError as exc:
                    if exc.code != 503:
                        raise
                except AppError:
                    if self.process.poll() is not None:
                        raise
                if healthy:
                    self.message = "Checking DNA tokenization and chat generation"
                    self.tokenizer_check()
                    self.inference_check()
                    if self.phase in ("stopping", "stopped"):
                        return
                    self.ready_at = time.time()
                    self.last_activity = time.monotonic()
                    self.phase, self.message = "ready", "Carbon-X Q8 · 8K context"
                    return
                time.sleep(2)
        except Exception as exc:
            reason = str(exc) if isinstance(exc, AppError) else f"Startup failed ({type(exc).__name__})"
            if self.process and self.process.poll() is not None:
                try:
                    print((self.model_dir / "carbon-model.log").read_text(errors="replace")[-3000:], flush=True)
                except OSError:
                    pass
            self.end(reason)

    def tokenizer_check(self):
        # The long canonical DNA segment must use 6-mers rather than text BPE.
        dna = "ACGTTGCAATGCCGTA" * 60
        obj = get_json(self.llama + "/tokenize", method="POST", data={"content": "<dna>" + dna + "</dna>", "add_special": False, "parse_special": True}, timeout=20)
        tokens = obj.get("tokens", [])
        if not isinstance(tokens, list) or not 155 <= len(tokens) <= 180:
            raise AppError("DNA tokenizer sanity check failed; refusing to serve")

    def inference_check(self):
        # Exercise the same chat-template/token-count path before showing Ready.
        payload = {"model": "carbon-x", "messages": [
            {"role": "user", "content": "Remember this test word: maple."},
            {"role": "assistant", "content": "The test word is maple."},
            {"role": "user", "content": "Repeat the test word, with no explanation."}],
            "max_tokens": 16, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
        counts = get_json(self.llama + "/v1/chat/completions/input_tokens", method="POST", data=payload, timeout=30)
        if not isinstance(counts, dict) or not 0 < finite_number(counts.get("input_tokens"), "prompt token count") < CONTEXT - 64:
            raise AppError("Chat token-count startup check failed")
        reply = get_json(self.llama + "/v1/chat/completions", method="POST", data=payload, timeout=90)
        choices = reply.get("choices") if isinstance(reply, dict) else None
        content = choices[0].get("message", {}).get("content") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if not isinstance(content, str) or not content.strip():
            raise AppError("Model produced no text in its startup chat check")

    def chat(self, handler, messages, role):
        if self.phase != "ready":
            return handler.respond(503, {"error": self.message})
        now = time.monotonic()
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        if len(self.requests) >= 20 or not self.generation.acquire(blocking=False):
            return handler.respond(429, {"error": "The GPU is busy. Please retry after the current reply."})
        self.requests.append(now)
        self.active += 1
        self.last_activity = now
        try:
            payload = {"model": "carbon-x", "messages": messages, "max_tokens": MAX_OUTPUT,
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0,
                       "chat_template_kwargs": {"enable_thinking": False}, "stream": True}
            counts = get_json(self.llama + "/v1/chat/completions/input_tokens", method="POST", data=payload, timeout=30)
            count = finite_number(counts.get("input_tokens"), "prompt token count")
            if count + MAX_OUTPUT + 32 > CONTEXT:
                return handler.respond(400, {"error": f"This conversation uses {int(count):,} input tokens. The 8K window reserves {MAX_OUTPUT:,} for the answer. Export this chat and start a shorter one; no history was discarded."})
            pipe_sse(handler, lambda: fetch(self.llama + "/v1/chat/completions", method="POST", data=payload, timeout=610))
        finally:
            self.active -= 1
            self.last_activity = time.monotonic()
            self.generation.release()


def worker_main():
    settings = json.loads(os.environ.pop("CARBON_SETTINGS"))
    api = Runpod(os.environ.pop("CARBON_RUNPOD_KEY"))
    os.environ.pop("CARBON_SOURCE", None)
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
        raise AppError("RUNPOD_POD_ID is missing; local launcher must clean up.")
    app = Worker(settings, api, pod_id)
    threading.Thread(target=app.watchdog, daemon=True).start()
    server = Server(("0.0.0.0", 8080), Handler)
    server.app = app
    threading.Thread(target=app.boot, daemon=True).start()
    server.serve_forever(poll_interval=0.5)


class Local:
    local = True
    guest = ""
    def __init__(self, directory, owner, api=None, ledger=None, entry=None, settings=None, demo=False):
        self.directory, self.owner, self.api = Path(directory), owner, api
        self.ledger, self.entry, self.settings = ledger, entry, settings or {}
        self.demo = demo
        self.phase, self.message = ("demo", "UI preview — no GPU or model") if demo else ("starting", "Runpod is starting the container")
        self.remote = None
        self.stopped = threading.Event()
        self.end_lock = threading.Lock()
        self.cache_lock = threading.Lock()
        self.cached = {}
        self.stop_reason = ""

    def history(self):
        return read_json(self.directory / "chats.json", {"chats": []}).get("chats", [])

    def save_history(self, data):
        chats = data.get("chats") if isinstance(data, dict) else None
        if not isinstance(chats, list) or len(chats) > 100:
            raise AppError("At most 100 saved chats are supported.")
        for c in chats:
            if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("title"), str) or not isinstance(c.get("messages"), list) or len(c["messages"]) > 256:
                raise AppError("Invalid conversation history.")
            for m in c["messages"]:
                if not isinstance(m, dict) or m.get("role") not in ("user", "assistant") or not isinstance(m.get("content"), str):
                    raise AppError("Invalid conversation message.")
        with self.cache_lock:
            atomic_json(self.directory / "chats.json", {"chats": chats})

    def status(self, role):
        with self.cache_lock:
            obj = dict(self.cached)
        obj.update({"local": True, "role": "owner", "phase": self.phase, "message": self.message})
        if self.settings:
            obj.update({"remaining_seconds": max(0, self.settings["deadline"] - time.time()), "hourly_rate": self.settings["rate"]})
        if self.settings.get("guest") and self.remote:
            obj["share_url"] = self.remote + "/#access=" + self.settings["guest"]
        if self.demo:
            obj["notice"] = "Preview mode. Responses are demonstrations; Carbon-X is not running."
        return obj

    def activity(self, role):
        if self.remote and self.phase == "ready":
            get_json(self.remote + "/api/activity", method="POST", data={}, token=self.owner, timeout=10)

    def chat(self, handler, messages, role):
        if self.demo:
            handler.sse_start()
            demo = "This is a UI preview, not a Carbon-X response. Your messages remain in the conversation. A real session sends the full saved history to the GPU on every turn."
            for word in demo.split(" "):
                handler.sse({"choices": [{"delta": {"content": word + " "}}]})
                time.sleep(0.025)
            handler.sse("[DONE]")
            return
        if self.phase != "ready" or not self.remote:
            return handler.respond(503, {"error": self.message})
        pipe_sse(handler, lambda: fetch(self.remote + "/api/chat", method="POST", data={"messages": messages}, token=self.owner, timeout=610), timeout=620)

    def end(self, reason):
        if not self.end_lock.acquire(blocking=False):
            return
        try:
            self.stop_reason = self.stop_reason or reason
            reason = self.stop_reason
            self.phase, self.message = "stopping", reason
            if self.demo:
                self.phase, self.message = "stopped", "Preview ended"
                self.stopped.set()
                return
            pod_id = self.entry.get("pod_id") if self.entry else None
            if not pod_id:
                raise AppError("Creation result is uncertain. Rerun with --stop to reconcile this installation's Pods.")
            for attempt in range(6):
                try:
                    self.api.delete(pod_id)
                    try:
                        self.ledger.close(self.entry)
                    except (OSError, AppError):
                        print("WARNING: Pod deletion confirmed, but the local spending ledger could not be saved. Preserve your state directory.", file=sys.stderr, flush=True)
                    self.phase, self.message = "stopped", reason + ". Pod termination confirmed; chats are saved."
                    self.stopped.set()
                    print(self.message, flush=True)
                    return
                except AppError:
                    if attempt == 5:
                        break
                    time.sleep(2)
            self.message = "Cleanup could not be confirmed. Check https://console.runpod.io/pods and run this script with --stop."
            print("WARNING: " + self.message, file=sys.stderr, flush=True)
        finally:
            self.end_lock.release()


def create_body(name, marker, image, choice, source, settings, key):
    bootstrap = "import os,base64,zlib;exec(compile(zlib.decompress(base64.b64decode(os.environ['CARBON_SOURCE'])), '<carbon_chat>', 'exec'))"
    return {"name": name, "cloud": choice["cloud"], "image": image,
            "gpu": {"id": choice["gpu"], "count": 1, "minRamPerGpu": 24,
                    "minVcpuCountPerGpu": 2, "minCudaVersion": MIN_CUDA_VERSION},
            "disk": DISK_GB, "mounts": {},
            "ports": ["8080/http"], "entrypoint": ["python3"], "cmd": ["-u", "-c", bootstrap],
            "env": {"CARBON_WORKER": "1", "CARBON_INSTALLATION": marker,
                    "CARBON_SOURCE": base64.b64encode(zlib.compress(source.encode(), 9)).decode(),
                    "CARBON_SETTINGS": json.dumps(settings), "CARBON_RUNPOD_KEY": key}}


def recover(api, ledger):
    # Do not infer ownership from a broad prefix: the name AND random marker match.
    pods = api.owned(ledger.name, ledger.data["installation"])
    for pod in pods:
        print("Removing this launcher's previous Pod " + pod["id"], flush=True)
        api.delete(pod["id"])
    for entry in ledger.data["sessions"]:
        if not entry.get("closed"):
            # Retain the full reservation for uncertain prior uptime, even if gone.
            entry["closed"] = True
            if time.time() > entry["deadline"]:
                entry["end"] = time.time() + 120 if any(p["id"] == entry.get("pod_id") for p in pods) else entry["deadline"]
    ledger.save()


def monitor(app):
    last_billing, last_inventory, failures, unavailable = 0.0, 0.0, 0, 0.0
    while not app.stopped.is_set():
        now = time.time()
        if now >= app.settings["deadline"]:
            app.end("Maximum session duration reached")
        if now >= app.settings["startup_deadline"] and app.phase == "starting":
            app.end("Startup timed out")
        if app.phase in ("stopping", "stopped"):
            if not app.stopped.is_set():
                app.end(app.stop_reason or "Retrying Pod cleanup")
            app.stopped.wait(15)
            continue
        try:
            status = get_json(app.remote + "/api/status", token=app.owner, timeout=8)
            if not isinstance(status, dict) or status.get("phase") not in ("starting", "ready", "stopping", "stopped"):
                raise AppError("Unexpected worker status.")
            with app.cache_lock:
                app.cached = status
            app.phase, app.message = status["phase"], status.get("message", "")
            unavailable = 0
            if app.phase in ("stopping", "stopped"):
                app.end(app.message or "Worker shutdown")
        except AppError:
            if not unavailable:
                unavailable = now
            if app.phase == "ready" and now - unavailable > 120:
                app.end("Worker unreachable for two minutes")
        if now - last_inventory >= 30 and not app.stopped.is_set():
            last_inventory = now
            try:
                pod = app.api.rest("/pods/" + app.entry["pod_id"])
                if not isinstance(pod, dict):
                    raise AppError("Unexpected Pod status response")
                if pod.get("status") not in ("PROVISIONING", "STARTING", "RUNNING"):
                    app.end("Pod is no longer running")
                elif actual_rate(pod) > app.settings["max_rate"]:
                    app.end("Pod price exceeds the approved ceiling")
            except HttpError as exc:
                if exc.code == 404:
                    app.end("Pod has ended")
            except AppError:
                pass
        if now - last_billing >= 120 and not app.stopped.is_set():
            last_billing = now
            try:
                reported = app.api.billing()
                if reconciled_spend(reported, app.ledger, live_entry=app.entry) >= ALLOWANCE - 0.10:
                    app.end("Monthly operating allowance reached")
                failures = 0
            except AppError:
                failures += 1
                if failures >= 2:
                    app.end("Cannot verify billing; ending session to limit spending")
        app.stopped.wait(3)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Launch Carbon-X chat on a disposable Runpod GPU. Python standard library only.", epilog="Budget: $15 operating allowance per UTC month, reserving $5 of a $20 budget. See this file's opening documentation for billing limitations.")
    p.add_argument("--version", action="version", version="Carbon-X launcher " + VERSION)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--stop", action="store_true", help="delete this installation's leftover Pods")
    mode.add_argument("--check", action="store_true", help="verify prices, model and image; do not create a Pod")
    mode.add_argument("--demo", action="store_true", help="preview the UI without cloud access or inference")
    p.add_argument("--share", action="store_true", help="enable a separate guest link while this Pod runs")
    p.add_argument("--max-gpu-price", type=float, default=0.40, help="maximum GPU USD/hour (default: 0.40)")
    p.add_argument("--idle-minutes", type=int, default=15, help="shutdown after no user activity (default: 15)")
    p.add_argument("--session-hours", type=float, default=4, help="maximum session including startup (default: 4)")
    p.add_argument("--port", type=int, default=8765, help="local browser UI port (default: 8765)")
    p.add_argument("--no-browser", action="store_true", help="print the local URL without opening a browser")
    p.add_argument("--state-dir", default=os.path.join(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")), "carbon-x-runpod"), help="local history and budget directory; keep it between sessions")
    args = p.parse_args(argv)
    if not math.isfinite(args.max_gpu_price) or not 0.01 <= args.max_gpu_price <= 0.60:
        p.error("--max-gpu-price must be between 0.01 and 0.60")
    if not math.isfinite(args.session_hours) or not 0.1 <= args.session_hours <= 8:
        p.error("--session-hours must be between 0.1 and 8")
    if not 1 <= args.idle_minutes <= 60 or not 1024 <= args.port <= 65535:
        p.error("--idle-minutes must be 1–60 and --port must be 1024–65535")
    return args


def local_main(argv=None):
    args = parse_args(argv)
    directory = Path(args.state_dir).expanduser().resolve()
    if args.demo:
        directory = directory / "preview"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    lock = open(directory / "launcher.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise AppError("This launcher is already running. Use its open chat window or stop it with Ctrl-C.") from None
    owner = secrets.token_urlsafe(32)
    # Bind before creating a paid resource. Never expose the owner's local UI to LAN.
    try:
        server = Server(("127.0.0.1", args.port), Handler) if not (args.stop or args.check) else None
    except OSError:
        lock.close()
        raise AppError(f"Local port {args.port} is busy. Rerun with --port 8766.") from None
    app, entry, api, ledger = None, None, None, None
    server_started = False
    creation_attempted = False
    try:
        if args.demo:
            app = Local(directory, owner, demo=True)
        else:
            key = os.environ.pop("RUNPOD_API_KEY", None) or getpass.getpass("Runpod API key (hidden): ")
            key = key.strip()
            if not key or any(c.isspace() for c in key):
                raise AppError("A Runpod API key is required.")
            api, ledger = Runpod(key), Ledger(directory)
            if args.stop:
                recover(api, ledger)
                print("Cleanup confirmed for this installation. Your local chats are preserved.")
                return 0
            if args.check:
                if api.owned(ledger.name, ledger.data["installation"]):
                    raise AppError("A previous Pod exists. Run --stop before another launch.")
            else:
                recover(api, ledger)
            print("Checking account balance, monthly spend, and live GPU prices…", flush=True)
            balance = api.account()
            reported = api.billing()
            used = reconciled_spend(reported, ledger)
            remaining = min(ALLOWANCE - used, balance - 0.25)
            if remaining < 0.25:
                raise AppError("Not enough remaining allowance or prepaid credit. No Pod created.")
            choices = api.choices(args.max_gpu_price)
            if args.check or not choices:
                api.describe_candidates(args.max_gpu_price)
            if not choices:
                raise AppError(f"No compatible GPU with at least 24 GB VRAM is reported available at or below ${args.max_gpu_price:.2f}/hour. No Pod created. See the catalog results above.")
            choice = choices[0]
            max_rate = args.max_gpu_price + DISK_RATE + 0.005
            if balance < max_rate:
                raise AppError("Runpod requires at least one hour of credit to deploy. Add credit within your budget.")
            print(f"Selected {choice['name']} ({choice['cloud'].lower()}): quoted GPU ${choice['rate']:.3f}/hour + estimated disk ${DISK_RATE:.3f}/hour.", flush=True)
            print(f"Monthly spend/reservations: ${used:.2f}; operating allowance remaining: ${max(0, ALLOWANCE-used):.2f}.", flush=True)
            print("Verifying the pinned model and a compatible official CUDA image…", flush=True)
            verify_model()
            image = resolve_image()
            if args.check:
                print("Preflight passed. No Pod created and no GPU charges incurred.")
                return 0
            start = time.time()
            _, month_end = month_bounds(start)
            duration = min(args.session_hours * 3600, (remaining - 0.10) / max_rate * 3600, month_end - start - 120)
            if duration < 600:
                raise AppError("Less than 10 minutes remain in the budget/month. No Pod created.")
            deadline = start + duration
            entry = ledger.reserve(start, deadline + 120, max_rate)
            settings = {"owner": owner, "guest": secrets.token_urlsafe(32) if args.share else "",
                        "installation": ledger.data["installation"], "deadline": deadline,
                        "startup_deadline": min(deadline, start + 1200), "idle_seconds": args.idle_minutes * 60,
                        "rate": choice["rate"] + DISK_RATE, "max_rate": max_rate}
            app = Local(directory, owner, api, ledger, entry, settings)
            source = Path(__file__).read_text()
            payload = create_body(ledger.name, ledger.data["installation"], image, choice, source, settings, key)
            print("Creating one on-demand Pod. Startup and model download are billable.", flush=True)
            print("The Pod watchdog receives the key privately so it can delete itself after disconnects.", flush=True)
            # Never retry this POST blindly: a timeout can occur after a successful creation.
            creation_attempted = True
            pod = api.rest("/pods", method="POST", data=payload)
            if not isinstance(pod, dict):
                raise AppError("Unexpected Pod creation response; reconciling by ownership marker.")
            pod_id = pod.get("id", "")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id):
                raise AppError("Runpod returned no usable Pod ID; reconciling by ownership marker.")
            entry["pod_id"] = pod_id
            ledger.save()
            if pod.get("mounts"):
                raise AppError("Runpod attached unexpected persistent storage; cleaning up the Pod.")
            rate = actual_rate(pod)
            if rate > max_rate:
                raise AppError(f"Assigned price ${rate:.3f}/hour exceeds the ${max_rate:.3f} ceiling. Cleaning up.")
            entry["rate"] = rate
            ledger.save()
            settings["rate"] = rate
            app.remote = f"https://{pod_id}-8080.proxy.runpod.net"
            print(f"Pod {pod_id}; conservative rate ${rate:.3f}/hour. Session ends within {duration/3600:.2f} hours.", flush=True)
            if args.share:
                print("Guest link: " + app.remote + "/#access=" + settings["guest"], flush=True)
            threading.Thread(target=monitor, args=(app,), daemon=True).start()
        server.app = app
        threading.Thread(target=server.serve_forever, daemon=True).start()
        server_started = True
        url = f"http://127.0.0.1:{args.port}/#access={owner}"
        print("Chat: " + url, flush=True)
        print("Keep this terminal running. Ctrl-C or End session removes the Pod. Chats stay on this computer.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        while not app.stopped.wait(0.5):
            pass
        # Keep the local window alive for export/history until the owner closes the launcher.
        print("GPU session is over. The local chat window remains available. Press Ctrl-C to close the launcher.", flush=True)
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nClosing session…", flush=True)
    finally:
        # Ctrl-C/SIGTERM can arrive after Runpod accepts POST but before the ID
        # reaches us. Reconcile this case too, without repeating the paid POST.
        if api and ledger and entry and creation_attempted and not entry.get("pod_id"):
            print("Reconciling uncertain creation before exiting…", flush=True)
            try:
                for attempt in range(3):
                    recover(api, ledger)
                    if attempt < 2:
                        time.sleep(1)
            except AppError:
                print("WARNING: Could not verify cleanup. Check Runpod Pods and run --stop before retrying.", file=sys.stderr)
        if app and not app.stopped.is_set() and (app.demo or (entry and entry.get("pod_id"))):
            app.end("Launcher closed")
        if server:
            if server_started:
                server.shutdown()
            server.server_close()
        lock.close()
    return 0


if __name__ == "__main__":
    try:
        if os.environ.get("CARBON_WORKER") == "1":
            worker_main()
        else:
            if sys.version_info < (3, 10):
                raise AppError("Python 3.10 or newer is required.")
            # SIGTERM follows the same cleanup path as Ctrl-C.
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            sys.exit(local_main())
    except AppError as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(1)
