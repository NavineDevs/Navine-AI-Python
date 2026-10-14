from __future__ import annotations

import re
from typing import Any, Dict

from navine.policy import http_get

IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def lookup_ip(ip: str, timeout: float = 10.0) -> Dict[str, Any]:
    value = (ip or "").strip()
    if not IP_RE.fullmatch(value):
        return {"ip": value, "valid": False, "error": "Invalid IPv4 address"}
    try:
        resp = http_get(
            f"http://ip-api.com/json/{value}?fields=status,message,country,regionName,city,zip,lat,lon,timezone,isp,org,as,query,reverse,mobile,proxy,hosting",
            timeout=timeout,
        )
        payload = resp.json()
        if payload.get("status") != "success":
            return {"ip": value, "valid": False, "error": payload.get("message") or "Lookup failed"}
        return {"ip": value, "valid": True, **payload}
    except Exception as exc:
        return {"ip": value, "valid": False, "error": str(exc)[:200]}


def format_ip_report(data: Dict[str, Any]) -> str:
    if not data.get("valid"):
        return f"IP lookup failed: {data.get('error') or 'invalid'}"
    lines = [
        f"IP intel `{data.get('ip')}`",
        "",
        f"Location: {data.get('city') or '?'}, {data.get('regionName') or '?'}, {data.get('country') or '?'}",
        f"ISP: {data.get('isp') or 'unknown'}",
        f"Org: {data.get('org') or 'unknown'}",
        f"ASN: {data.get('as') or 'unknown'}",
        f"Timezone: {data.get('timezone') or 'unknown'}",
    ]
    if data.get("reverse"):
        lines.append(f"Reverse DNS: {data.get('reverse')}")
    flags = []
    if data.get("proxy"):
        flags.append("proxy/VPN")
    if data.get("hosting"):
        flags.append("hosting/datacenter")
    if data.get("mobile"):
        flags.append("mobile")
    if flags:
        lines.append(f"Flags: {', '.join(flags)}")
    return "\n".join(lines).strip()
