from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from navine.policy import http_get

COMMON_EMAIL_DOMAINS = (
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "proton.me",
    "protonmail.com",
    "icloud.com",
    "aol.com",
    "mail.com",
    "gmx.com",
)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)


def _clean_handle(value: str) -> str:
    text = str(value or "").strip().lstrip("@")
    text = re.sub(r"[^A-Za-z0-9._\-]", "", text)
    return text[:64]


def guess_emails_from_username(username: str) -> List[str]:
    user = _clean_handle(username).lower()
    if not user or len(user) < 2:
        return []
    base = user.replace("-", ".").replace("__", ".")
    alts = {user, base, base.replace(".", ""), base.replace(".", "_")}
    if "." in base:
        parts = [p for p in base.split(".") if p]
        if len(parts) >= 2:
            alts.add(f"{parts[0]}.{parts[-1]}")
            alts.add(f"{parts[0]}{parts[-1]}")
    emails: List[str] = []
    for local in sorted(alts):
        if not re.fullmatch(r"[a-z0-9._%+\-]{2,48}", local):
            continue
        for domain in COMMON_EMAIL_DOMAINS[:6]:
            emails.append(f"{local}@{domain}")
    return emails[:18]


def check_gravatar(email: str) -> Optional[Dict[str, Any]]:
    addr = str(email or "").strip().lower()
    if not EMAIL_RE.fullmatch(addr):
        return None
    digest = hashlib.md5(addr.encode("utf-8")).hexdigest()
    url = f"https://www.gravatar.com/avatar/{digest}?d=404&s=80"
    try:
        response = http_get(url, timeout=8)
        ctype = str(response.headers.get("Content-Type") or "").lower()
        if response.status_code == 200 and "image" in ctype:
            return {
                "email": addr,
                "gravatar": True,
                "profile_url": f"https://www.gravatar.com/{digest}",
                "avatar_url": url.replace("d=404", "d=identicon"),
            }
    except Exception:
        return None
    return None


def extract_emails_and_websites(texts: List[str]) -> Dict[str, List[str]]:
    emails: List[str] = []
    websites: List[str] = []
    seen_e = set()
    seen_w = set()
    for raw in texts:
        text = str(raw or "")
        for match in EMAIL_RE.findall(text):
            low = match.lower()
            if low not in seen_e:
                seen_e.add(low)
                emails.append(low)
        for match in URL_RE.findall(text):
            cleaned = match.rstrip(").,;]'\"")
            host = (urlparse(cleaned).hostname or "").lower()
            if not host or host in seen_w:
                continue
            seen_w.add(host)
            websites.append(cleaned)
    return {"emails": emails[:20], "websites": websites[:20]}


def check_password_pwned(password: str) -> Dict[str, Any]:
    secret = str(password or "")
    if len(secret) < 4:
        return {"checked": False, "reason": "password too short to check"}
    digest = hashlib.sha1(secret.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]
    try:
        response = http_get(
            f"https://api.pwnedpasswords.com/range/{prefix}",
            timeout=12,
            headers={"Add-Padding": "true", "User-Agent": "Navine-OSINT"},
        )
        text = response.text or ""
        count = 0
        for line in text.splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 2 and parts[0].upper() == suffix:
                count = int(parts[1].replace(",", ""))
                break
        return {
            "checked": True,
            "breached": count > 0,
            "seen_count": count,
            "method": "hibp_k_anonymity",
            "note": "Checks public breach corpora only. No password is stored.",
        }
    except Exception as exc:
        return {"checked": False, "reason": str(exc)[:200]}


def check_email_breaches(email: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    addr = str(email or "").strip().lower()
    if not EMAIL_RE.fullmatch(addr):
        return {"checked": False, "reason": "invalid email"}
    key = str(api_key or "").strip()
    if not key:
        return {
            "checked": False,
            "email": addr,
            "reason": "HIBP API key not configured",
            "hint": "Set osint.hibp_api_key to enable account breach lookup.",
        }
    try:
        response = http_get(
            f"https://haveibeenpwned.com/api/v3/breachedaccount/{addr}?truncateResponse=true",
            timeout=15,
            headers={"hibp-api-key": key, "User-Agent": "Navine-OSINT"},
        )
        if response.status_code == 404:
            return {"checked": True, "email": addr, "breached": False, "breach_count": 0, "breaches": []}
        rows = response.json() if response.content else []
        names = [str(row.get("Name") or row) for row in rows] if isinstance(rows, list) else []
        return {
            "checked": True,
            "email": addr,
            "breached": bool(names),
            "breach_count": len(names),
            "breaches": names[:25],
        }
    except Exception as exc:
        return {"checked": False, "email": addr, "reason": str(exc)[:240]}


def guess_profile_websites(username: str) -> List[str]:
    user = _clean_handle(username)
    if not user:
        return []
    templates = (
        "https://github.com/{u}",
        "https://gitlab.com/{u}",
        "https://www.reddit.com/user/{u}",
        "https://x.com/{u}",
        "https://twitter.com/{u}",
        "https://instagram.com/{u}",
        "https://www.tiktok.com/@{u}",
        "https://www.youtube.com/@{u}",
        "https://www.linkedin.com/in/{u}",
        "https://{u}.github.io",
        "https://{u}.wordpress.com",
        "https://medium.com/@{u}",
        "https://keybase.io/{u}",
        "https://about.me/{u}",
    )
    return [t.format(u=user) for t in templates]


def enrich_username(username: str, profile_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    user = _clean_handle(username)
    texts: List[str] = [user]
    found_urls: List[str] = []
    for row in profile_rows or []:
        url = str(row.get("url") or "").strip()
        site = str(row.get("site") or "").strip()
        if url:
            found_urls.append(url)
            texts.append(url)
        if site:
            texts.append(site)
    extracted = extract_emails_and_websites(texts)
    guessed = guess_emails_from_username(user)
    profile_guesses = guess_profile_websites(user)
    for url in profile_guesses:
        if url not in extracted["websites"]:
            extracted["websites"].append(url)
    gravatar_hits: List[Dict[str, Any]] = []
    for email in guessed[:8]:
        hit = check_gravatar(email)
        if hit:
            gravatar_hits.append(hit)
            if email not in extracted["emails"]:
                extracted["emails"].append(email)
    for url in found_urls:
        if url not in extracted["websites"]:
            extracted["websites"].append(url)
    return {
        "username": user,
        "valid": bool(user),
        "candidate_emails": guessed[:12],
        "confirmed_emails": [h["email"] for h in gravatar_hits],
        "gravatar": gravatar_hits,
        "emails": extracted["emails"][:20],
        "websites": extracted["websites"][:30],
        "password_note": (
            "To check if a password appeared in public breaches, ask: "
            "check password <value> (HIBP k-anonymity, nothing stored)."
        ),
    }


def format_email_intel_report(payload: Dict[str, Any]) -> str:
    lines = ["Email / identity enrichment", ""]
    emails = list(payload.get("emails") or [])
    confirmed = list(payload.get("confirmed_emails") or [])
    websites = list(payload.get("websites") or [])
    candidates = list(payload.get("candidate_emails") or [])
    if confirmed:
        lines.append("Gravatar-linked emails:")
        for email in confirmed:
            lines.append(f"- {email}")
        lines.append("")
    if emails:
        lines.append("Emails discovered or linked:")
        for email in emails[:12]:
            lines.append(f"- {email}")
        lines.append("")
    if candidates and not confirmed:
        lines.append("Common email patterns to verify manually:")
        for email in candidates[:8]:
            lines.append(f"- {email}")
        lines.append("")
    if websites:
        lines.append("Websites / profile URLs:")
        for url in websites[:12]:
            lines.append(f"- {url}")
        lines.append("")
    breach = payload.get("breach")
    if isinstance(breach, dict) and breach.get("checked"):
        if breach.get("breached"):
            names = ", ".join(breach.get("breaches") or [])
            lines.append(f"Breach hits: {breach.get('breach_count', 0)} public corpora ({names}).")
        else:
            lines.append("Breach hits: none reported for this email via configured API.")
        lines.append("")
    pwd = payload.get("password_check")
    if isinstance(pwd, dict) and pwd.get("checked"):
        if pwd.get("breached"):
            lines.append(
                f"Password check: appeared in public dumps about {pwd.get('seen_count', 0)} times."
            )
        else:
            lines.append("Password check: not found in the HIBP public dump range.")
        lines.append("")
    note = str(payload.get("password_note") or "").strip()
    if note:
        lines.append(note)
    if len(lines) <= 2:
        lines.append("No email or website enrichment hits yet.")
    return "\n".join(lines).strip()
