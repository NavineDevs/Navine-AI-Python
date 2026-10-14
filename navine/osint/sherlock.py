from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from navine.policy import http_get

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{2,32}$")

PROBE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class SherlockSite:
    name: str
    url: str
    ok_codes: Tuple[int, ...] = (200,)
    bad_codes: Tuple[int, ...] = (404,)
    exists_text: Optional[str] = None
    missing_text: Optional[str] = None
    allow_redirects: bool = True


SHERLOCK_SITES: Tuple[SherlockSite, ...] = (
    SherlockSite("GitHub", "https://github.com/{user}"),
    SherlockSite("GitLab", "https://gitlab.com/{user}"),
    SherlockSite("Bitbucket", "https://bitbucket.org/{user}/"),
    SherlockSite("Codeberg", "https://codeberg.org/{user}"),
    SherlockSite("Gitea", "https://gitea.com/{user}"),
    SherlockSite("Reddit", "https://www.reddit.com/user/{user}"),
    SherlockSite("Dev.to", "https://dev.to/{user}"),
    SherlockSite("HackerRank", "https://www.hackerrank.com/{user}"),
    SherlockSite("Replit", "https://replit.com/@{user}"),
    SherlockSite("Keybase", "https://keybase.io/{user}"),
    SherlockSite("Medium", "https://medium.com/@{user}"),
    SherlockSite("GitHub Gist", "https://gist.github.com/{user}"),
    SherlockSite("Steam", "https://steamcommunity.com/id/{user}", bad_codes=(404, 403)),
    SherlockSite("Twitch", "https://www.twitch.tv/{user}"),
    SherlockSite("YouTube", "https://www.youtube.com/@{user}"),
    SherlockSite("TikTok", "https://www.tiktok.com/@{user}"),
    SherlockSite("Pinterest", "https://www.pinterest.com/{user}/"),
    SherlockSite("SoundCloud", "https://soundcloud.com/{user}"),
    SherlockSite("Vimeo", "https://vimeo.com/{user}"),
    SherlockSite("Flickr", "https://www.flickr.com/people/{user}/"),
    SherlockSite("Behance", "https://www.behance.net/{user}"),
    SherlockSite("Dribbble", "https://dribbble.com/{user}"),
    SherlockSite("Spotify", "https://open.spotify.com/user/{user}"),
    SherlockSite("Last.fm", "https://www.last.fm/user/{user}"),
    SherlockSite("Discogs", "https://www.discogs.com/user/{user}"),
    SherlockSite("Docker Hub", "https://hub.docker.com/u/{user}"),
    SherlockSite("npm", "https://www.npmjs.com/~{user}"),
    SherlockSite("PyPI", "https://pypi.org/user/{user}/"),
    SherlockSite("RubyGems", "https://rubygems.org/profiles/{user}"),
    SherlockSite("Packagist", "https://packagist.org/users/{user}/"),
    SherlockSite("Hacker News", "https://news.ycombinator.com/user?id={user}"),
    SherlockSite("Product Hunt", "https://www.producthunt.com/@{user}"),
    SherlockSite("About.me", "https://about.me/{user}"),
    SherlockSite("Gravatar", "https://en.gravatar.com/{user}"),
    SherlockSite("TryHackMe", "https://tryhackme.com/p/{user}"),
    SherlockSite("HackTheBox", "https://app.hackthebox.com/profile/{user}"),
    SherlockSite("Pastebin", "https://pastebin.com/u/{user}"),
    SherlockSite("Telegram", "https://t.me/{user}"),
    SherlockSite("X", "https://x.com/{user}", missing_text="This account does not exist"),
    SherlockSite(
        "Instagram",
        "https://www.instagram.com/{user}/",
        missing_text="Sorry, this page isn't available",
    ),
    SherlockSite("Facebook", "https://www.facebook.com/{user}", bad_codes=(404, 400)),
    SherlockSite("LinkedIn", "https://www.linkedin.com/in/{user}/", bad_codes=(404, 999)),
    SherlockSite("Snapchat", "https://www.snapchat.com/add/{user}"),
    SherlockSite("Roblox", "https://www.roblox.com/user.aspx?username={user}"),
    SherlockSite("Chess.com", "https://www.chess.com/member/{user}"),
    SherlockSite("Duolingo", "https://www.duolingo.com/profile/{user}"),
    SherlockSite("Mastodon.social", "https://mastodon.social/@{user}"),
)


def clean_username(value: str) -> str:
    return re.sub(r"^[@#]+", "", (value or "").strip())


def _classify_response(site: SherlockSite, status: int, body: str) -> str:
    text = (body or "")[:12000].lower()
    if site.missing_text and site.missing_text.lower() in text:
        return "not_found"
    if site.exists_text and site.exists_text.lower() in text:
        return "found"
    if status in site.bad_codes:
        return "not_found"
    if status in site.ok_codes:
        return "found"
    if 300 <= status < 400:
        return "found"
    if status in (401, 403, 429, 999):
        return "unknown"
    return "unknown"


def probe_site(site: SherlockSite, username: str, timeout: float = 10.0) -> Dict[str, Any]:
    url = site.url.format(user=username)
    started = time.perf_counter()
    try:
        response = http_get(
            url,
            headers=PROBE_HEADERS,
            timeout=timeout,
            allow_redirects=site.allow_redirects,
        )
        elapsed = int((time.perf_counter() - started) * 1000)
        body = response.text or ""
        state = _classify_response(site, int(response.status_code), body)
        return {
            "site": site.name,
            "url": url,
            "status": state,
            "http_status": int(response.status_code),
            "response_ms": elapsed,
        }
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        return {
            "site": site.name,
            "url": url,
            "status": "error",
            "http_status": 0,
            "response_ms": elapsed,
            "error": str(exc)[:180],
        }


def scan_username(
    username: str,
    max_workers: int = 12,
    timeout: float = 10.0,
    max_sites: Optional[int] = None,
) -> Dict[str, Any]:
    user = clean_username(username)
    if not user or not USERNAME_RE.fullmatch(user):
        return {
            "username": user,
            "valid": False,
            "profiles": [],
            "found": [],
            "not_found": [],
            "unknown": [],
            "errors": [],
            "summary": {"total": 0, "found": 0, "not_found": 0, "unknown": 0, "errors": 0},
        }

    sites = list(SHERLOCK_SITES)
    if max_sites is not None:
        sites = sites[: max(1, int(max_sites))]

    profiles: List[Dict[str, Any]] = []
    workers = max(1, min(int(max_workers), 16))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(probe_site, site, user, timeout): site for site in sites}
        for future in as_completed(futures):
            try:
                profiles.append(future.result())
            except Exception as exc:
                site = futures[future]
                profiles.append(
                    {
                        "site": site.name,
                        "url": site.url.format(user=user),
                        "status": "error",
                        "http_status": 0,
                        "response_ms": 0,
                        "error": str(exc)[:180],
                    }
                )

    profiles.sort(key=lambda row: (row.get("status") != "found", row.get("site") or ""))
    found = [row for row in profiles if row.get("status") == "found"]
    not_found = [row for row in profiles if row.get("status") == "not_found"]
    unknown = [row for row in profiles if row.get("status") == "unknown"]
    errors = [row for row in profiles if row.get("status") == "error"]
    return {
        "username": user,
        "valid": True,
        "profiles": profiles,
        "found": found,
        "not_found": not_found,
        "unknown": unknown,
        "errors": errors,
        "summary": {
            "total": len(profiles),
            "found": len(found),
            "not_found": len(not_found),
            "unknown": len(unknown),
            "errors": len(errors),
        },
    }


def format_sherlock_report(scan: Dict[str, Any]) -> str:
    if not scan.get("valid"):
        return "Invalid username. Use 2-32 characters: letters, numbers, underscore, dot, hyphen."
    user = scan.get("username") or ""
    summary = scan.get("summary") or {}
    lines = [
        f"Username scan `{user}`",
        "",
        f"Checked {summary.get('total', 0)} public sites — "
        f"**{summary.get('found', 0)} found**, "
        f"{summary.get('not_found', 0)} not found, "
        f"{summary.get('unknown', 0)} inconclusive.",
        "",
    ]
    found = scan.get("found") or []
    if found:
        lines.append("Accounts likely found:")
        for row in found:
            ms = row.get("response_ms")
            extra = f" ({ms} ms)" if ms else ""
            lines.append(f"- **{row.get('site')}**: {row.get('url')}{extra}")
        lines.append("")
    unknown = scan.get("unknown") or []
    if unknown:
        lines.append("Manual check recommended (blocked or ambiguous):")
        for row in unknown[:8]:
            lines.append(f"- {row.get('site')}: {row.get('url')}")
        if len(unknown) > 8:
            lines.append(f"- ... and {len(unknown) - 8} more")
        lines.append("")
    if not found:
        lines.append("No confirmed profiles on the scanned sites. Try web search below or a different spelling.")
    lines.append("Note: Public pages only. False positives/negatives happen when sites block bots or hide profiles.")
    return "\n".join(lines).strip()
