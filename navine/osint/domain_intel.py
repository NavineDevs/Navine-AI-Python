from __future__ import annotations

import socket
from typing import Any, Dict
from urllib.parse import urlparse

from navine.policy import http_get


def _clean_domain(value: str) -> str:
    text = (value or "").strip().lower()
    if "://" in text:
        text = urlparse(text).netloc or text
    if text.startswith("www."):
        text = text[4:]
    return text.split("/")[0].split(":")[0]


def lookup_domain(domain: str, timeout: float = 8.0) -> Dict[str, Any]:
    host = _clean_domain(domain)
    if not host or "." not in host:
        return {"domain": host, "valid": False, "error": "Invalid domain"}

    records: Dict[str, Any] = {"domain": host, "valid": True, "a": [], "aaaa": [], "mx": [], "ns": []}
    try:
        _, _, addrs = socket.gethostbyname_ex(host)
        records["a"] = sorted(set(addrs))
    except socket.gaierror as exc:
        records["a_error"] = str(exc)

    try:
        import dns.resolver

        resolver = dns.resolver.Resolver()
        resolver.lifetime = timeout
        for rtype, key in (("A", "a"), ("AAAA", "aaaa"), ("MX", "mx"), ("NS", "ns")):
            try:
                answers = resolver.resolve(host, rtype)
                if rtype == "MX":
                    records[key] = sorted(
                        {f"{row.preference} {row.exchange}".strip() for row in answers}
                    )
                else:
                    records[key] = sorted({str(row).rstrip(".") for row in answers})
            except Exception:
                continue
    except ImportError:
        records["dns_note"] = "Install dnspython for full MX/NS lookup: pip install dnspython"

    try:
        resp = http_get(f"https://{host}", timeout=timeout, allow_redirects=True)
        records["http_status"] = int(resp.status_code)
        records["final_url"] = str(resp.url)
        records["server"] = (resp.headers.get("Server") or "").strip()
    except Exception as exc:
        records["http_error"] = str(exc)[:200]

    return records


def format_domain_report(data: Dict[str, Any]) -> str:
    if not data.get("valid"):
        return f"Could not parse domain: {data.get('error') or 'invalid'}"
    lines = [f"Domain intel `{data.get('domain')}`", ""]
    if data.get("a"):
        lines.append("A records:")
        for row in data["a"]:
            lines.append(f"- {row}")
    if data.get("aaaa"):
        lines.append("")
        lines.append("AAAA records:")
        for row in data["aaaa"]:
            lines.append(f"- {row}")
    if data.get("mx"):
        lines.append("")
        lines.append("MX records:")
        for row in data["mx"]:
            lines.append(f"- {row}")
    if data.get("ns"):
        lines.append("")
        lines.append("NS records:")
        for row in data["ns"]:
            lines.append(f"- {row}")
    if data.get("http_status"):
        lines.append("")
        lines.append(f"HTTPS: {data.get('http_status')} -> {data.get('final_url') or data.get('domain')}")
        if data.get("server"):
            lines.append(f"Server: {data.get('server')}")
    if data.get("a_error") and not data.get("a"):
        lines.append("")
        lines.append(f"DNS lookup failed: {data.get('a_error')}")
    if data.get("dns_note"):
        lines.append("")
        lines.append(str(data.get("dns_note")))
    return "\n".join(lines).strip()
