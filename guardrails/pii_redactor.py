"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.
Patterns covered: person names (spaCy NER + regex fallback), usernames/login IDs,
email addresses, employee IDs, IP addresses, and phone numbers.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text, known_identifiers=["jsmith01"])
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)
"""

import os
import re
import json
from datetime import datetime

# Try to import spaCy — graceful fallback if not installed
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except (ImportError, OSError):
    SPACY_AVAILABLE = False
    print("⚠  spaCy not available — using regex-only PII detection "
          "(run: pip install spacy && python -m spacy download en_core_web_sm)")

# ── REGEX PATTERNS ────────────────────────────────────────────────────────────
# Order matters: emails/IDs are masked first so the username and name rules
# never grab pieces of them. Where a pattern has a named group "pii", only that
# group is masked (the label word such as "username:" is kept for context).
# Ticket refs (INC/REQ/CHG...) are deliberately NOT masked — they are not PII.

# Label words that are followed by a username ("username jsmith", "user id: x")
_ID_KW = (r"(?:user\s*-?name|user\s*-?id|log(?:in|on)\s*-?(?:name|id)|"
          r"account\s*-?(?:name|id)|sam\s*account\s*name|network\s*id|ad\s*id|"
          r"windows\s*id|uid)")
# Short label words, only trusted with a ':' '=' '#' separator ("user: jsmith")
_BARE_KW = r"(?:user|login|logon|account)"
# Words that follow "username" in normal sentences but are not usernames
_NOT_A_USER = r"(?!(?:and|or|is|was|the|for|of|to|not|has|have|field|change|reset|password)\b)"
_USER_TOKEN = r"[A-Za-z][\w.\-]*\w"

PATTERNS = [
    ("EMAIL",       re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("EMPLOYEE_ID", re.compile(r"\b(?:EMP|ZEN)-?\d{3,6}\b", re.I)),
    ("IP_ADDRESS",  re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    # (?<!\w) instead of \b so the "+91-" prefix is masked too
    ("PHONE",       re.compile(r"(?<![\w+])(?:\+91[\-\s]?)?\d{10}\b|\b\d{3}[\-\s]\d{3}[\-\s]\d{4}\b")),

    # USERNAME 1: DOMAIN\user   e.g. ZENSAR\pdengale
    ("USERNAME",    re.compile(r"(?P<pii>\b[A-Za-z][A-Za-z0-9\-]{1,14}\\[A-Za-z][\w.\-]{0,63}\w)")),
    # USERNAME 2: labelled      e.g. "username: jsmith", "login id is r.kumar", "user id jsmith"
    ("USERNAME",    re.compile(rf"\b{_ID_KW}\s*(?:[:=#]|\bis\b)?\s*{_NOT_A_USER}(?P<pii>{_USER_TOKEN})", re.I)),
    ("USERNAME",    re.compile(rf"\b{_BARE_KW}\s*[:=#]\s*(?P<pii>{_USER_TOKEN})", re.I)),
    # USERNAME 3: username-shaped word after a label, e.g. "user jsmith01", "account p.dengale"
    ("USERNAME",    re.compile(rf"\b(?i:{_BARE_KW}|{_ID_KW})\s+"
                               r"(?P<pii>[a-z][a-z0-9]*(?:[._\-][a-z][a-z0-9]*)+|[a-z]{2,}\d{1,4})\b")),
]

# Regex fallback for names when spaCy is missing: two or three Capitalised words
# after a cue word ("User John Smith", "for Michael D'Souza", "caller Priya Nair").
_NAME_FALLBACK = re.compile(
    r"\b(?i:user|for|by|caller|contact|employee|contractor|requester|mr\.?|ms\.?|mrs\.?|dr\.?)\s+"
    r"(?P<pii>[A-Z][a-z]+(?:[ \-][A-Z](?:'[A-Z])?[a-z']+){1,2})\b"
)

# ── AUDIT LOGGER ──────────────────────────────────────────────────────────────

audit_log = []

def _audit(action, detail):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "module": "PIIRedactor",
        "action": action,
        "detail": detail
    }
    audit_log.append(entry)
    return entry

# ── REDACTION FUNCTION ────────────────────────────────────────────────────────

def _is_acronym(text: str) -> bool:
    """spaCy sometimes tags short acronyms (SLA, VPN, KB) as PERSON.
    Only skip single short ALL-CAPS words — 'JOHN SMITH' from a legacy
    export is still a real name and must be masked."""
    return text.isupper() and " " not in text and len(text) <= 5


def redact(text: str, known_identifiers: list[str] | None = None) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1]
      - mapping: dict to restore original values later

    known_identifiers: usernames you already know belong to this ticket
      (e.g. caller_id / opened_by from ServiceNow). Plain lowercase usernames
      like "jsmith" are impossible to tell apart from ordinary words by pattern
      alone, so passing them here is the most reliable way to catch them.

    Example:
      clean, m = redact("Contact john.doe@corp.com or call 9876543210")
      # clean  = "Contact [EMAIL_1] or call [PHONE_1]"
      # m      = {"[EMAIL_1]": "john.doe@corp.com", "[PHONE_1]": "9876543210"}
    """
    mapping = {}    # token -> original
    reverse = {}    # original -> token (same value always gets the same token)
    counters = {}

    def token_for(label, value):
        if value in reverse:
            return reverse[value]
        counters[label] = counters.get(label, 0) + 1
        tok = f"[{label}_{counters[label]}]"
        mapping[tok] = value
        reverse[value] = tok
        return tok

    def mask(label, regex, s):
        def repl(m):
            g = "pii" if "pii" in regex.groupindex else 0
            start, end = m.start(g) - m.start(), m.end(g) - m.start()
            whole = m.group(0)
            return whole[:start] + token_for(label, m.group(g)) + whole[end:]
        return regex.sub(repl, s)

    clean = text

    # Step 1: Regex patterns (emails, IDs, IPs, phones, usernames)
    for label, regex in PATTERNS:
        clean = mask(label, regex, clean)

    # Step 2: Usernames supplied by the caller (ticket metadata)
    for ident in sorted({i for i in (known_identifiers or []) if i}, key=len, reverse=True):
        rx = re.compile(rf"(?<![\w.\\\[]){re.escape(ident)}(?![\w@\]])", re.I)
        clean = rx.sub(lambda m: token_for("USERNAME", m.group(0)), clean)

    # Step 3: Person names — spaCy NER, or regex fallback if spaCy is missing
    if SPACY_AVAILABLE:
        for ent in nlp(clean).ents:
            name = ent.text.strip()
            if ent.label_ != "PERSON" or "[" in name or "]" in name or _is_acronym(name):
                continue
            tok = token_for("NAME", name)
            # whole-word replace so "Ram" doesn't hit "Ramesh" or "program"
            clean = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", lambda _m: tok, clean)
    else:
        clean = mask("NAME", _NAME_FALLBACK, clean)

    pii_count = len(mapping)
    if pii_count > 0:
        _audit("redact", f"{pii_count} PII item(s) masked: {list(mapping.keys())}")
    else:
        _audit("redact", "No PII detected")

    return clean, mapping

def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token, original in mapping.items():
        restored = restored.replace(token, original)
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored

def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────

class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval.
    The rationale is free text written by agents, so it is redacted before it is
    written — otherwise the audit file itself becomes a PII leak."""

    def __init__(self, log_file: str = "logs/audit_trail.jsonl"):
        os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A",
            known_identifiers: list[str] | None = None,
            timestamp: str | None = None, echo: bool = True):
        safe_rationale, pii = redact(rationale, known_identifiers) if rationale else ("", {})
        entry = {
            "timestamp": timestamp or datetime.now().isoformat(),   # keep the node's own time if given
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale[:200],
            "pii_masked": len(pii),
            "approval_status": approval_status
        }
        self.entries.append(entry)

        # Append to JSONL file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        if echo:
            print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'='*55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'='*55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("PII REDACTION DEMO")
    print("=" * 55)

    sample_tickets = [
        ("User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.", None),
        ("Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.", None),
        ("Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.", None),
        ("VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.", None),
        ("Account locked for user jsmith01. Username: pdengale, domain login ZENSAR\\pdengale.", None),
        ("Please unlock account p.dengale — user is unable to log in. Username and password reset needed.", None),
        ("Reset MFA for rkumar, he is travelling. Ticket INC0001005.", ["rkumar"]),
    ]

    for i, (ticket, known) in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket, known_identifiers=known)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")
        assert restore(clean, mapping) == ticket, "restore() did not round-trip"

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger("logs/demo_audit.jsonl")
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 210 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to user jsmith01 at john.smith@zensar.com — auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: logs/demo_audit.jsonl")
    print(f"Last rationale on disk: {logger.entries[-1]['rationale']}")