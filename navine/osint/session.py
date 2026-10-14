from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from navine.osint.domain_intel import format_domain_report, lookup_domain
from navine.osint.email_intel import (
    check_email_breaches,
    check_password_pwned,
    enrich_username,
    format_email_intel_report,
)
from navine.osint.ip_intel import format_ip_report, lookup_ip
from navine.osint.sherlock import clean_username, format_sherlock_report, scan_username


def _build_search_query(target: str, kind: str) -> str:
    text = str(target or "").strip()
    key = str(kind or "auto").strip().lower()
    if not text:
        return ""
    if key == "domain":
        return f"{text} WHOIS DNS records subdomains site history"
    if key == "ip":
        return f"{text} IP geolocation ASN reverse DNS reputation"
    if key == "username":
        return (
            f'"{text}" username email contact website profile '
            f"site:github.com OR site:reddit.com OR site:linkedin.com"
        )
    if key == "email":
        return f"{text} breach leak paste public mentions domain owner"
    if key == "password":
        return f"password breach check public dump hygiene guidance"
    if key == "person":
        return f'"{text}" public profile professional records news'
    if key == "organization":
        return f"{text} company registration leadership public filings news"
    if key == "phone":
        return f"{text} carrier region public listings"
    return f"{text} open source intelligence public records"


def _format_search_hits(results: List[Dict[str, Any]]) -> str:
    lines = []
    for row in results[:10]:
        title = str(row.get("title") or "").strip()
        snippet = str(row.get("snippet") or row.get("body") or row.get("excerpt") or "").strip()
        url = str(row.get("url") or "").strip()
        if not (title or snippet or url):
            continue
        chunk = title
        if snippet:
            chunk = f"{chunk}\n{snippet}" if chunk else snippet
        if url:
            chunk = f"{chunk}\nSource: {url}"
        lines.append(chunk[:1200])
    return "\n\n".join(lines)


def username_profile_links(username: str) -> List[str]:
    user = clean_username(username)
    scan = scan_username(user, max_workers=8, max_sites=12)
    found = scan.get("found") or []
    if found:
        return [f"- {row.get('site')}: {row.get('url')}" for row in found]
    profiles = scan.get("profiles") or []
    return [f"- {row.get('site')}: {row.get('url')}" for row in profiles[:12]]


OSINT_TARGET_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "my", "your", "our", "their",
    "how", "what", "when", "where", "why", "who", "whom", "which", "many", "much",
    "total", "sum", "difference", "product", "average", "number", "amount", "cost",
    "price", "apples", "oranges", "people", "person", "users", "user", "me", "you",
    "him", "her", "them", "it", "and", "or", "for", "from", "with", "into", "onto",
    "about", "after", "before", "more", "less", "left", "right", "first", "second",
    "third", "each", "every", "all", "some", "any", "few", "several", "then", "than",
    "solve", "answer", "question", "problem", "math", "calculate", "find",
}


def detect_osint_kind(message: str, target: Optional[str] = None) -> str:
    lower = (message or "").lower()
    if re.search(r"\b(?:username|user\s*name|handle|social\s+profile|sherlock)\b", lower):
        return "username"
    if re.search(r"\b(?:password|passwd|pwned\s+password|breach(?:ed)?\s+password)\b", lower):
        return "password"
    if target and "@" in target:
        return "email"
    if target and re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", target) and "@" not in target and "." not in target:
        if re.search(r"\b(?:username|handle|profile|social|sherlock|email|website|breach)\b", lower):
            return "username"
    if re.search(r"\b(?:domain|dns|whois|subdomain)\b", lower):
        return "domain"
    if re.search(r"\b(?:email|breach)\b", lower):
        return "email"
    if re.search(r"\b(?:ip address|ip lookup|trace ip)\b", lower):
        return "ip"
    if re.search(r"\b(?:phone|carrier)\b", lower):
        return "phone"
    if re.search(r"\b(?:company|organization|org)\b", lower):
        return "organization"
    if re.search(r"\b(?:person|people search)\b", lower):
        return "person"
    if target:
        if re.fullmatch(r"[\w.+-]+@[\w.-]+\.\w+", target):
            return "email"
        if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", target):
            return "ip"
        if "." in target and "@" not in target:
            return "domain"
        if re.fullmatch(r"[A-Za-z0-9_.-]{2,32}", target):
            return "username"
    return "auto"


def extract_osint_target(message: str) -> Tuple[Optional[str], str]:
    text = (message or "").strip()
    if not text:
        return None, "auto"

    try:
        from navine.realtime.math import is_math_query
        from navine.text.quality import needs_osint

        if is_math_query(text) and not needs_osint(text):
            return None, "auto"
    except Exception:
        pass

    quoted = re.search(r'["\']([^"\']{2,120})["\']', text)
    if quoted:
        target = quoted.group(1).strip()
        if target.lower() not in OSINT_TARGET_STOPWORDS:
            return target, detect_osint_kind(text, target)

    patterns = [
        r"(?i)(?:username|handle|sherlock)\s*[:=]?\s*([A-Za-z0-9_.-]{2,32})",
        r"(?i)(?:look(?:ing)?\s+(?:up|for)|find|search(?:ing)?\s+for|trace|investigate|scan)\s+username\s+([A-Za-z0-9_.-]{2,32})",
        r"(?i)(?:domain|site|website)\s+([a-z0-9][a-z0-9.-]+\.[a-z]{2,})",
        r"(?i)(?:email)\s+([\w.+-]+@[\w.-]+\.\w+)",
        r"([\w.+-]+@[\w.-]+\.\w+)",
        r"(?i)(?:ip(?:\s+address)?)\s+((?:\d{1,3}\.){3}\d{1,3})",
        r"\b((?:\d{1,3}\.){3}\d{1,3})\b",
    ]
    try:
        from navine.text.quality import needs_osint

        strong_osint = needs_osint(text)
    except Exception:
        strong_osint = False
    if strong_osint:
        patterns.insert(
            1,
            r"(?i)(?:look(?:ing)?\s+(?:up|for)|find|search(?:ing)?\s+for|trace|investigate|scan)\s+(?:the\s+)?(?:username|handle|user)\s+([A-Za-z0-9_.-]{2,32})",
        )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            target = match.group(1).strip()
            if target.lower() in OSINT_TARGET_STOPWORDS:
                continue
            if target.isdigit():
                continue
            return target, detect_osint_kind(text, target)
    return None, detect_osint_kind(text)


def is_osint_capability_only(message: str) -> bool:
    from navine.search.web import detect_capability_kind

    if detect_capability_kind(message) == "osint":
        return True
    lower = (message or "").lower()
    if re.search(r"\bcan\s+(?:you|it)\s+help.{0,40}username\b", lower):
        if not extract_osint_target(message)[0]:
            return True
    if re.search(r"\b(?:look|search|find|sherlock)\s+(?:for\s+)?(?:a\s+)?username\b", lower):
        if not extract_osint_target(message)[0]:
            return True
    return False


def format_osint_report(payload: Dict[str, Any]) -> str:
    target = str(payload.get("target") or "").strip()
    kind = str(payload.get("kind") or "auto").strip().lower()
    results = list(payload.get("search_results") or [])
    sections: List[str] = [f"OSINT report for `{target}` ({kind})", ""]

    username_scan = payload.get("username_scan")
    if kind == "username" and username_scan:
        sections.append(format_sherlock_report(username_scan))
        sections.append("")

    email_intel = payload.get("email_intel")
    if email_intel:
        sections.append(format_email_intel_report(email_intel))
        sections.append("")

    domain_intel = payload.get("domain_intel")
    if kind == "domain" and domain_intel:
        sections.append(format_domain_report(domain_intel))
        sections.append("")

    ip_intel = payload.get("ip_intel")
    if kind == "ip" and ip_intel:
        sections.append(format_ip_report(ip_intel))
        sections.append("")

    if kind == "password" and payload.get("password_check"):
        sections.append(format_email_intel_report({"password_check": payload.get("password_check")}))
        sections.append("")

    if results:
        sections.append("Public web findings:")
        for idx, row in enumerate(results[:10], start=1):
            title = str(row.get("title") or f"Result {idx}").strip()
            url = str(row.get("url") or "").strip()
            snippet = str(row.get("snippet") or row.get("excerpt") or row.get("body") or "").strip()
            sections.append(f"{idx}. {title}")
            if url:
                sections.append(f"   {url}")
            if snippet:
                sections.append(f"   {snippet[:260]}")
        sections.append("")
        sections.append("Confidence: medium from public sources only. Verify each link manually.")
    elif kind not in {"username", "domain", "ip", "email", "password"} or not (
        username_scan or domain_intel or ip_intel or email_intel or payload.get("password_check")
    ):
        sections.append("No live web hits were returned right now.")
        sections.append("Try the OSINT tab with public web search enabled, or rephrase with the exact target.")

    if kind == "username":
        sections.append(
            "Next steps: open found profiles, verify candidate emails, follow website links, "
            "then check public breach corpora for confirmed emails."
        )
    elif kind == "email":
        sections.append("Next steps: confirm mailbox ownership signals, check MX, and review public breach corpora.")
    elif kind == "password":
        sections.append("Next steps: if breached, change it everywhere and enable MFA. Never reuse passwords.")
    elif kind == "domain":
        sections.append("Next steps: check certificate transparency, historical DNS, and public breach/news mentions.")
    elif kind == "ip":
        sections.append("Next steps: verify ASN ownership, check reverse DNS, and search for public blocklist reports.")

    return "\n".join(sections).strip()


def _analysis_is_usable(text: str) -> bool:
    lower = (text or "").lower()
    if not text or len(text.strip()) < 24:
        return False
    blocked = (
        "here is my best direct take",
        "tell me what you want in more detail",
        "open the osint tab",
        "look up username example_user",
    )
    return not any(marker in lower for marker in blocked)


def investigate(
    target: str,
    kind: str = "auto",
    use_search: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    subject = str(target or "").strip()
    if not subject:
        raise ValueError("Target is required")
    kind_key = str(kind or "auto").strip().lower() or "auto"
    if kind_key == "auto":
        kind_key = detect_osint_kind("", subject)

    username_scan: Optional[Dict[str, Any]] = None
    domain_intel: Optional[Dict[str, Any]] = None
    ip_intel: Optional[Dict[str, Any]] = None
    email_intel: Optional[Dict[str, Any]] = None
    password_check: Optional[Dict[str, Any]] = None

    hibp_key = None
    try:
        from navine.utils.config import load_config

        hibp_key = str((load_config("navine").get("osint") or {}).get("hibp_api_key") or "").strip() or None
    except Exception:
        hibp_key = None

    if kind_key == "username":
        try:
            username_scan = scan_username(subject, max_workers=12)
        except Exception:
            username_scan = None
        found_rows = []
        if username_scan:
            found_rows = list(username_scan.get("found") or username_scan.get("profiles") or [])
        try:
            email_intel = enrich_username(subject, found_rows)
        except Exception:
            email_intel = None
    elif kind_key == "email":
        try:
            email_intel = enrich_username(subject.split("@")[0], [{"url": f"mailto:{subject}"}])
            email_intel["emails"] = list(dict.fromkeys([subject.lower()] + list(email_intel.get("emails") or [])))
            breach = check_email_breaches(subject, api_key=hibp_key)
            email_intel["breach"] = breach
            grav = None
            try:
                from navine.osint.email_intel import check_gravatar

                grav = check_gravatar(subject)
            except Exception:
                grav = None
            if grav:
                email_intel.setdefault("gravatar", []).append(grav)
                email_intel["confirmed_emails"] = list(
                    dict.fromkeys(list(email_intel.get("confirmed_emails") or []) + [subject.lower()])
                )
        except Exception:
            email_intel = None
    elif kind_key == "password":
        password_check = check_password_pwned(subject)
    elif kind_key == "domain":
        try:
            domain_intel = lookup_domain(subject)
        except Exception:
            domain_intel = None
    elif kind_key == "ip":
        try:
            ip_intel = lookup_ip(subject)
        except Exception:
            ip_intel = None

    query = _build_search_query(subject, kind_key)
    search_results: List[Dict[str, Any]] = []
    search_context = ""
    if use_search and query and kind_key != "password":
        try:
            from navine.search.web import search_internet

            search_results = search_internet(query, max_results=10) or []
            search_context = _format_search_hits(search_results)
            if search_results:
                try:
                    from navine.learn.from_search import learn_from_search_results

                    learn_from_search_results(query, search_results)
                except Exception:
                    pass
        except Exception:
            search_results = []
            search_context = ""

    if kind_key == "username" and email_intel is not None and search_results:
        try:
            from navine.osint.email_intel import extract_emails_and_websites

            extra = extract_emails_and_websites(
                [
                    f"{row.get('title') or ''} {row.get('snippet') or row.get('body') or ''} {row.get('url') or ''}"
                    for row in search_results
                ]
            )
            for email in extra.get("emails") or []:
                if email not in email_intel["emails"]:
                    email_intel["emails"].append(email)
            for url in extra.get("websites") or []:
                if url not in email_intel["websites"]:
                    email_intel["websites"].append(url)
        except Exception:
            pass

    has_structured = bool(
        (username_scan and username_scan.get("valid"))
        or (domain_intel and domain_intel.get("valid"))
        or (ip_intel and ip_intel.get("valid"))
        or (email_intel and email_intel.get("valid"))
        or password_check
        or search_results
    )

    analysis = ""
    payload_preview = {
        "target": subject if kind_key != "password" else "[redacted-password]",
        "kind": kind_key,
        "query": query,
        "search_results": search_results[:10],
        "username_scan": username_scan,
        "domain_intel": domain_intel,
        "ip_intel": ip_intel,
        "email_intel": email_intel,
        "password_check": password_check,
    }
    if has_structured:
        analysis = format_osint_report(payload_preview)
    elif search_context:
        prompt = (
            f"Target: {subject}\n"
            f"Investigation type: {kind_key}\n"
            "Perform lawful open-source intelligence only.\n"
            "Summarize verified public findings, cite sources, note confidence, and list next research steps.\n"
            "Do not provide illegal access, doxxing, or unauthorized intrusion steps.\n\n"
            f"Public web findings:\n{search_context[:6000]}"
        )
        try:
            from navine.text.chat import _run_generation

            raw = _run_generation(
                prompt,
                None,
                None,
                search_context,
                False,
                460,
                0.3,
                osint_mode=True,
            )
            if _analysis_is_usable(raw):
                analysis = raw.strip()
        except Exception:
            analysis = ""

    payload = {
        "target": subject if kind_key != "password" else "[redacted-password]",
        "kind": kind_key,
        "query": query,
        "analysis": analysis,
        "search_results": search_results[:10],
        "username_scan": username_scan,
        "domain_intel": domain_intel,
        "ip_intel": ip_intel,
        "email_intel": email_intel,
        "password_check": password_check,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    if not payload["analysis"]:
        payload["analysis"] = format_osint_report(payload)

    out_root = Path(output_dir) if output_dir else Path("outputs") / "osint"
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_src = subject if kind_key != "password" else "password_check"
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in safe_src)[:48] or "target"
    manifest = out_root / f"report_{safe}_{stamp}.json"
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    payload["report_path"] = str(manifest)
    return payload
