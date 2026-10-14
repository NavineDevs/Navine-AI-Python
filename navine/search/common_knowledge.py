import re
from typing import Dict, Optional, Tuple

CAPITALS: Dict[str, str] = {
    "afghanistan": "Kabul",
    "albania": "Tirana",
    "algeria": "Algiers",
    "argentina": "Buenos Aires",
    "australia": "Canberra",
    "austria": "Vienna",
    "bangladesh": "Dhaka",
    "belgium": "Brussels",
    "brazil": "Brasilia",
    "bulgaria": "Sofia",
    "cambodia": "Phnom Penh",
    "canada": "Ottawa",
    "chile": "Santiago",
    "china": "Beijing",
    "colombia": "Bogota",
    "croatia": "Zagreb",
    "cuba": "Havana",
    "czech republic": "Prague",
    "czechia": "Prague",
    "denmark": "Copenhagen",
    "egypt": "Cairo",
    "ethiopia": "Addis Ababa",
    "finland": "Helsinki",
    "france": "Paris",
    "germany": "Berlin",
    "greece": "Athens",
    "hungary": "Budapest",
    "iceland": "Reykjavik",
    "india": "New Delhi",
    "indonesia": "Jakarta",
    "iran": "Tehran",
    "iraq": "Baghdad",
    "ireland": "Dublin",
    "israel": "Jerusalem",
    "italy": "Rome",
    "japan": "Tokyo",
    "kenya": "Nairobi",
    "malaysia": "Kuala Lumpur",
    "mexico": "Mexico City",
    "morocco": "Rabat",
    "netherlands": "Amsterdam",
    "new zealand": "Wellington",
    "nigeria": "Abuja",
    "north korea": "Pyongyang",
    "norway": "Oslo",
    "pakistan": "Islamabad",
    "peru": "Lima",
    "philippines": "Manila",
    "poland": "Warsaw",
    "portugal": "Lisbon",
    "romania": "Bucharest",
    "russia": "Moscow",
    "saudi arabia": "Riyadh",
    "scotland": "Edinburgh",
    "serbia": "Belgrade",
    "singapore": "Singapore",
    "south africa": "Pretoria",
    "south korea": "Seoul",
    "spain": "Madrid",
    "sweden": "Stockholm",
    "switzerland": "Bern",
    "thailand": "Bangkok",
    "turkey": "Ankara",
    "ukraine": "Kyiv",
    "united arab emirates": "Abu Dhabi",
    "uae": "Abu Dhabi",
    "united kingdom": "London",
    "uk": "London",
    "england": "London",
    "united states": "Washington, D.C.",
    "united states of america": "Washington, D.C.",
    "usa": "Washington, D.C.",
    "us": "Washington, D.C.",
    "vietnam": "Hanoi",
    "wales": "Cardiff",
}

CAPITAL_QUERY = re.compile(
    r"(?:what(?:'s|\s+is)\s+)?(?:the\s+)?capital\s+(?:city\s+)?of\s+(.+?)[\s?.!]*$",
    re.I,
)

COMMON_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    (
        re.compile(
            r"\bcicada\s*3301\b"
            r"|\b(?:are you able|can you|could you|do you)\s+(?:to\s+)?(?:solve|crack|decode)\s+cicada\b"
            r"|\bwhat\s+is\s+cicada(?:\s*3301)?\b"
            r"|\btell\s+me\s+about\s+cicada(?:\s*3301)?\b",
            re.I,
        ),
        (
            "No complete public solution exists for Cicada 3301 as a whole. "
            "It was a series of internet puzzles (2012–2014) mixing crypto, steganography, and real-world clues; "
            "early rounds were solved by community solvers, but later stages and the group's full purpose stay unresolved. "
            "I can explain known history, discuss public puzzle techniques at a high level, or help with a specific cipher you paste—"
            "I cannot magically finish the unsolved Liber Primus or claim a secret final answer."
        ),
    ),
    (
        re.compile(
            r"\bwhat\s+(?:colour|color)s?\s+(?:is|are)\s+(?:the\s+)?sky\b"
            r"|\bwhat(?:'s|\s+is)\s+the\s+(?:colour|color)\s+of\s+(?:the\s+)?sky\b"
            r"|\bsky\s+(?:colour|color)\b",
            re.I,
        ),
        (
            "During a clear day the sky usually looks blue because air scatters shorter blue "
            "wavelengths of sunlight more than longer ones (Rayleigh scattering). "
            "At sunrise and sunset it can look orange, red, pink, or purple because sunlight "
            "travels through more atmosphere and more blue light is scattered away. "
            "Cloudy skies look gray or white when many water droplets scatter light in all directions."
        ),
    ),
    (
        re.compile(
            r"\bwhat\s+(?:colour|color)\s+(?:is|are)\s+(?:the\s+)?grass\b"
            r"|\bwhat(?:'s|\s+is)\s+the\s+(?:colour|color)\s+of\s+(?:the\s+)?grass\b",
            re.I,
        ),
        "Healthy grass usually looks green because chlorophyll reflects green light and absorbs other wavelengths for photosynthesis.",
    ),
    (
        re.compile(
            r"\bwhat\s+(?:colour|color)\s+(?:is|are)\s+(?:the\s+)?(?:sun|sunlight)\b",
            re.I,
        ),
        (
            "Sunlight is a mix of many colors that together look white to our eyes. "
            "Through the atmosphere the Sun often looks yellow-white, and redder near the horizon."
        ),
    ),
    (
        re.compile(r"\bwhy\s+is\s+(?:the\s+)?sky\s+blue\b", re.I),
        (
            "The sky looks blue because molecules in the air scatter short blue wavelengths of sunlight "
            "more strongly than red ones. That scattered blue light reaches your eyes from many directions."
        ),
    ),
    (
        re.compile(r"\b(?:explain|what\s+is)\s+recursion\b", re.I),
        "Recursion is when a function solves a problem by calling itself on a smaller version of the same problem until it reaches a simple base case.",
    ),
    (
        re.compile(r"\bwhat\s+is\s+an?\s+api\b", re.I),
        "An API is a defined way for one program to request data or actions from another program without sharing all of its internal code.",
    ),
    (
        re.compile(r"\bwhat\s+is\s+a\s+variable\b", re.I),
        "A variable is a named place in memory that stores a value so a program can read or change it later.",
    ),
    (
        re.compile(r"\bwhat\s+is\s+machine\s+learning\b", re.I),
        "Machine learning is a way for programs to learn patterns from data instead of hard-coding every rule. Models train on examples and then make predictions on new inputs.",
    ),
    (
        re.compile(r"\bwhat\s+is\s+water\s+made\s+of\b", re.I),
        "Water is H2O, two hydrogen atoms bonded to one oxygen atom.",
    ),
    (
        re.compile(r"\bexplain\s+photosynthesis\b|\bwhat\s+is\s+photosynthesis\b", re.I),
        "Photosynthesis is how plants make food from sunlight, water, and carbon dioxide. They produce glucose for energy and release oxygen as a byproduct.",
    ),
    (
        re.compile(r"\b(?:summarize|explain|what\s+is)\s+(?:a\s+)?loop\b", re.I),
        "A loop repeats a block of code until a condition ends it. Common forms are for-loops over items and while-loops that continue while a condition stays true.",
    ),
    (
        re.compile(
            r"\bhow\s+to\s+(?:make|build|assemble|manufacture)\s+(?:a\s+)?(?:smart\s*)?phones?\b",
            re.I,
        ),
        (
            "Making a phone is a large industrial process, not a home DIY project. At a high level: "
            "1) Define the product (screen size, battery, radios, OS, cost). "
            "2) Design the electronics: application processor, memory, power management, modem, antennas, cameras, sensors, and PCB layout. "
            "3) Design the mechanical enclosure, display stack, and thermal path. "
            "4) Write or adapt firmware and an operating system that drive the hardware. "
            "5) Prototype, test RF/certification, battery safety, drop/thermal reliability, then manufacture at scale with supply-chain partners. "
            "Individuals usually buy components or modify existing devices; full phone manufacturing needs factories, FCC/CE-style certification, and specialized tooling."
        ),
    ),
    (
        re.compile(
            r"\bhow\s+to\s+(?:make|build)\s+(?:a\s+)?computers?\b",
            re.I,
        ),
        (
            "To assemble a typical PC: choose a compatible CPU, motherboard, RAM, storage, power supply, and case; "
            "optionally add a GPU and cooler. Mount the motherboard, install CPU and RAM, connect storage and PSU cables, "
            "then install an operating system and drivers. Building from raw silicon chips is factory-level work."
        ),
    ),
    (
        re.compile(
            r"\b(?:what\s+is|what'?s|explain(?:\s+what)?|define)\b.{0,24}\bgpus?\b|"
            r"\bgpus?\b.{0,24}\b(?:do|does|for|used\s+for)\b",
            re.I,
        ),
        (
            "A GPU (graphics processing unit) is a processor built for highly parallel math. "
            "It renders graphics and also accelerates AI, video, and scientific workloads by running many simple operations at once."
        ),
    ),
    (
        re.compile(
            r"\bhow\s+to\s+(?:make|bake)\s+(?:a\s+)?(?:bread|loaf)\b",
            re.I,
        ),
        (
            "Basic bread: mix flour, water, yeast, and salt; knead until elastic; let it rise until doubled; "
            "shape the loaf; proof again; bake until the crust is brown and the interior is set. Cool before slicing."
        ),
    ),
)


def _normalize_place(raw: str) -> str:
    text = re.sub(r"[^\w\s]", " ", (raw or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+(country|nation|republic|kingdom)$", "", text)
    return text.strip()


def answer_capital_query(message: str) -> Optional[str]:
    stripped = (message or "").strip()
    if not stripped:
        return None
    match = CAPITAL_QUERY.search(stripped)
    if not match:
        return None
    place = _normalize_place(match.group(1))
    if not place:
        return None
    capital = CAPITALS.get(place)
    if capital:
        display = place.title() if place not in ("uk", "uae", "usa", "us") else place.upper()
        if place == "us":
            display = "the United States"
        elif place == "uk":
            display = "the United Kingdom"
        elif place == "uae":
            display = "the UAE"
        elif place == "usa":
            display = "the United States"
        return f"The capital of {display} is {capital}."
    return None


def is_common_knowledge_query(message: str) -> bool:
    stripped = (message or "").strip()
    if not stripped:
        return False
    if answer_capital_query(stripped):
        return True
    for pattern, _ in COMMON_PATTERNS:
        if pattern.search(stripped):
            return True
    return False


def answer_common_knowledge(message: str) -> Optional[str]:
    stripped = (message or "").strip()
    if not stripped:
        return None
    capital = answer_capital_query(stripped)
    if capital:
        return capital
    for pattern, answer in COMMON_PATTERNS:
        if pattern.search(stripped):
            return answer
    return None
