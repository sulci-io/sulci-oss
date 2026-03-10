"""
benchmark/run.py
================
10,000-query semantic cache benchmark for Sulci.

Runs entirely without API keys or cloud accounts.
Uses a built-in TF-IDF cosine similarity engine to simulate
sentence-transformer embeddings — no ML dependencies required.

Produces (in benchmark/results/):
  summary.json            — overall stats
  domain_breakdown.csv    — per-domain hit rates and cost savings
  threshold_sweep.csv     — hit rate vs threshold (0.70 → 0.95)
  time_series.csv         — hit rate evolution over time
  false_positives.csv     — near-miss analysis

Usage:
  # Standalone (no install needed)
  python benchmark/run.py

  # With real sulci embeddings (better accuracy)
  pip install "sulci[sqlite]"
  python benchmark/run.py --use-sulci

Options:
  --use-sulci     Use sulci.Cache with SQLite backend instead of built-in engine
  --threshold N   Similarity threshold (default: 0.85)
  --queries N     Number of test queries to run (default: 5000)
  --no-sweep      Skip threshold sweep (faster)
  --out DIR       Output directory (default: benchmark/results)
"""

import argparse
import csv
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional

random.seed(42)

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Sulci benchmark")
parser.add_argument("--use-sulci",  action="store_true",
                    help="Use sulci.Cache (requires pip install 'sulci[sqlite]')")
parser.add_argument("--threshold",  type=float, default=0.85)
parser.add_argument("--queries",    type=int,   default=5000,
                    help="Number of test queries (warmup is equal)")
parser.add_argument("--no-sweep",   action="store_true")
parser.add_argument("--out",        default=os.path.join(
                        os.path.dirname(__file__), "results"))
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1.  BUILT-IN EMBEDDING ENGINE
#     TF-IDF cosine similarity — no external dependencies.
#     Approximates sentence-transformer paraphrase detection on short queries.
# ══════════════════════════════════════════════════════════════════════════════

VOCAB = (
    "what is how the a of to for in can do you make use build why when where who "
    "best way difference between explain tell me about does work example simple create "
    "get set run start stop help need want should would could will show find fix error "
    "problem issue code data model api llm ai machine learning cache semantic vector "
    "embedding query response system database server python javascript function class "
    "method return value type string number list array key search index similarity "
    "threshold cosine dot product normalize dimension token text language natural "
    "processing nlp transformer bert llama fine tune train inference prompt completion "
    "generate output input context memory retrieval augmented generation rag chunk "
    "document knowledge base store reduce cost latency speed fast slow performance "
    "optimize save money expensive cheap free open source cancel subscription billing "
    "account password reset login logout update change delete remove add new order "
    "return refund shipping delivery track status payment invoice address phone email "
    "support contact hours policy terms privacy security feature bug deploy release "
    "install configure setup environment variable cloud aws azure docker container "
    "kubernetes microservice architecture design pattern test debug log monitor alert "
    "dashboard metric analytics report export import format parse validate schema "
    "migrate backup restore version upgrade rollback dependency package library "
    "framework react vue angular typescript interface component state hook effect "
    "async await promise callback event listener handler middleware route endpoint "
    "request response header body status auth token jwt oauth sql nosql query join "
    "index foreign key transaction commit rollback cursor aggregate pipeline filter "
    "sort limit skip project match group unwind lookup medical diagnosis treatment "
    "symptom prescription dosage side effect drug interaction patient doctor hospital "
    "appointment insurance coverage claim deductible premium copay referral specialist "
    "emergency urgent care pharmacy lab test result scan xray mri blood pressure "
    "diabetes heart disease cancer vaccine allergy chronic acute infection antibiotic "
    "legal contract clause liability warranty disclaimer intellectual property patent "
    "trademark copyright license agreement terms conditions dispute arbitration court "
    "compliance regulation gdpr hipaa sox audit risk assessment mitigation control"
).split()

VOCAB_IDX = {w: i for i, w in enumerate(VOCAB)}
DIM = len(VOCAB)


def _tokenize(text: str) -> list:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()


def _embed(text: str) -> list:
    tokens = _tokenize(text)
    tf: dict = defaultdict(float)
    for t in tokens:
        if t in VOCAB_IDX:
            tf[VOCAB_IDX[t]] += 1.0
    if not tf:
        return [0.0] * DIM
    vec = [0.0] * DIM
    for idx, cnt in tf.items():
        vec[idx] = 1 + math.log(cnt)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


# ── Built-in cache (LSH-accelerated, no deps) ─────────────────────────────────

@dataclass
class _Entry:
    query:    str
    response: str
    vec:      list
    group:    str = ""
    domain:   str = ""


class _BuiltinCache:
    N_PROJ = 16

    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.entries: list[_Entry] = []
        self.hits = self.misses = 0
        rng = random.Random(42)
        self._proj = []
        for _ in range(self.N_PROJ):
            v    = [rng.gauss(0, 1) for _ in range(DIM)]
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            self._proj.append([x / norm for x in v])
        self._buckets: dict = {}

    def _lsh(self, vec: list) -> int:
        bits = 0
        for i, p in enumerate(self._proj):
            if sum(a * b for a, b in zip(vec, p)) > 0:
                bits |= (1 << i)
        return bits

    def get(self, query: str) -> tuple:
        qv = _embed(query)
        h  = self._lsh(qv)
        candidates: set = set()
        if h in self._buckets:
            candidates.update(self._buckets[h])
        for i in range(self.N_PROJ):
            nh = h ^ (1 << i)
            if nh in self._buckets:
                candidates.update(self._buckets[nh])
        best_sim, best_entry = 0.0, None
        for idx in candidates:
            e   = self.entries[idx]
            sim = _cosine(qv, e.vec)
            if sim > best_sim:
                best_sim, best_entry = sim, e
        if best_sim >= self.threshold:
            self.hits += 1
            return best_entry.response, best_sim, best_entry
        self.misses += 1
        return None, best_sim, None

    def set(self, query: str, response: str, group: str = "", domain: str = "") -> None:
        vec = _embed(query)
        idx = len(self.entries)
        self.entries.append(_Entry(query, response, vec, group, domain))
        h = self._lsh(vec)
        self._buckets.setdefault(h, []).append(idx)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SULCI CACHE WRAPPER  (optional, --use-sulci flag)
# ══════════════════════════════════════════════════════════════════════════════

class _SulciWrapper:
    """Thin wrapper around sulci.Cache to match the built-in interface."""

    def __init__(self, threshold: float, db_path: str):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        try:
            from sulci import Cache
        except ImportError:
            print("ERROR: sulci not installed. Run: pip install \"sulci[sqlite]\"")
            sys.exit(1)
        self._cache   = Cache(backend="sqlite", threshold=threshold,
                              db_path=db_path, ttl_seconds=None)
        self.threshold = threshold
        self.entries: list = []   # dummy — used for group lookups
        self._group_map: dict = {}
        self.hits = self.misses = 0

    def get(self, query: str) -> tuple:
        response, sim = self._cache.get(query)
        if response is not None:
            self.hits += 1
            # find closest entry for correctness check
            matched = self._group_map.get(response)
            return response, sim, matched
        self.misses += 1
        return None, sim, None

    def set(self, query: str, response: str, group: str = "", domain: str = "") -> None:
        self._cache.set(query, response)
        self._group_map[response] = type("E", (), {"group": group, "domain": domain})()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  QUERY CORPUS  (5 domains × 10 topic groups × ~100 queries each)
# ══════════════════════════════════════════════════════════════════════════════

DOMAINS = {
    "customer_support": {
        "templates": [
            ("cancel subscription",   ["How do I cancel my subscription?", "I want to cancel my account", "Cancel my plan please", "How to stop my subscription", "Unsubscribe from service", "Cancel renewal of my plan", "How do I stop being charged?", "I need to cancel my membership", "Cancel account request", "Steps to cancel subscription"]),
            ("reset password",        ["How do I reset my password?", "I forgot my password", "Cannot log in, password help", "Password reset instructions", "How to change my password?", "Lost password recovery", "Help I can't access my account", "Reset login credentials", "Forgot account password", "How to recover password?"]),
            ("refund policy",         ["What is your refund policy?", "Can I get a refund?", "How do I request a refund?", "Money back guarantee?", "Return and refund process", "I want my money back", "Refund request procedure", "How long does refund take?", "Is there a refund option?", "Refund eligibility criteria"]),
            ("update billing",        ["How do I update my billing info?", "Change credit card on file", "Update payment method", "New card for billing", "How to change billing address?", "Update my payment details", "Switch payment method", "Add new payment card", "Billing information update", "Change my card details"]),
            ("track order",           ["Where is my order?", "Track my shipment", "Order tracking information", "When will my order arrive?", "Check delivery status", "Shipping status for my order", "How to track my package?", "Order not delivered yet", "Delivery tracking help", "Where is my package?"]),
            ("contact support",       ["How do I contact support?", "Customer service phone number", "How to reach help desk?", "Support contact information", "Get help from customer service", "Speak to a representative", "Contact customer care", "Support hours and availability", "How to talk to someone?", "Reach the support team"]),
            ("change email",          ["How do I change my email address?", "Update account email", "Change email on my account", "New email address setup", "How to update login email?", "Change registered email", "Email address change request", "Update contact email", "Modify account email", "Change my account email address"]),
            ("account locked",        ["My account is locked", "Cannot access locked account", "Account suspended help", "How to unlock my account?", "Account access blocked", "Locked out of account", "Account deactivated issue", "Reactivate locked account", "Why is my account locked?", "Unlock account assistance"]),
            ("upgrade plan",          ["How do I upgrade my plan?", "Upgrade to premium", "Switch to higher tier plan", "Upgrade account plan", "How to get premium features?", "Plan upgrade instructions", "Move to a better plan", "Upgrade my subscription tier", "Higher plan benefits", "Upgrade from basic to pro"]),
            ("invoice download",      ["How to download my invoice?", "Get billing receipt", "Download payment invoice", "Where are my invoices?", "Invoice download instructions", "Access billing history", "Get my receipt or invoice", "Billing documents download", "How to get my invoice?", "Download past invoices"]),
        ],
        "responses": {
            "cancel": "To cancel, go to Account Settings > Subscription > Cancel Plan. Access continues until end of billing period.",
            "reset":  "Click 'Forgot Password' on the login page. You will receive a reset email within 5 minutes.",
            "refund": "Refunds allowed within 30 days. Processed within 5-7 business days to original payment method.",
            "billing":"Update billing at Account Settings > Billing > Payment Methods.",
            "track":  "Track at Orders > Track Shipment. Real-time updates sent via email.",
            "contact":"Reach us at support@company.com or 1-800-SUPPORT. Mon-Fri 9am-6pm EST.",
            "email":  "Change email at Account Settings > Profile > Email Address. Verification required.",
            "locked": "Account locked after failed attempts. Click 'Unlock Account' in the email we sent.",
            "upgrade":"Upgrade at Account Settings > Subscription > Change Plan. Effective immediately, prorated.",
            "invoice":"Download invoices at Account Settings > Billing > Invoice History.",
        },
    },
    "developer_qa": {
        "templates": [
            ("async await python",    ["How does async await work in Python?", "Python asyncio explanation", "Async functions in Python tutorial", "What is asyncio in Python?", "How to use await in Python", "Python async programming guide", "Asynchronous Python code example", "Understanding async def in Python", "Python coroutines explained", "How to write async code Python"]),
            ("git merge vs rebase",   ["What is the difference between git merge and rebase?", "Git rebase vs merge explained", "When to use git rebase vs merge", "Merge or rebase in git?", "Git merge vs rebase differences", "Should I use rebase or merge git?", "Explain git rebase versus merge", "Git history rebase vs merge", "Rebase vs merge which is better?", "Git branching merge vs rebase"]),
            ("docker container",      ["How do Docker containers work?", "Explain Docker containers", "What is a Docker container?", "Docker containers tutorial", "How to use Docker containers", "Docker container vs image", "Getting started with Docker", "Docker basics for beginners", "What does Docker container do?", "Docker containerisation explained"]),
            ("react usestate hook",   ["How does useState work in React?", "React useState hook explained", "Using useState in React components", "What is useState React hook?", "React state management with hooks", "useState example in React", "How to use React useState", "State management useState React", "React functional component state", "useState hook tutorial React"]),
            ("sql vs nosql",          ["What is the difference between SQL and NoSQL?", "SQL vs NoSQL databases comparison", "When to use NoSQL vs SQL?", "Relational vs non-relational database", "SQL NoSQL differences explained", "Choose SQL or NoSQL database", "NoSQL vs SQL which is better?", "Database SQL versus NoSQL", "Comparing SQL and NoSQL databases", "SQL NoSQL pros and cons"]),
            ("rest api design",       ["How do I design a REST API?", "REST API best practices", "RESTful API design principles", "Building a good REST API", "REST API design guidelines", "How to design RESTful endpoints", "REST API architecture explained", "Principles of REST API design", "Good practices for REST APIs", "REST API design patterns"]),
            ("python list comprehension", ["How do list comprehensions work in Python?", "Python list comprehension syntax", "List comprehension examples Python", "What are Python list comprehensions?", "Using list comprehension in Python", "Python comprehension explained", "List comprehension vs for loop Python", "Python list comprehension guide", "How to write list comprehension", "Python list comprehension tutorial"]),
            ("kubernetes deployment", ["How do Kubernetes deployments work?", "Kubernetes deployment explained", "What is a Kubernetes deployment?", "Deploy application on Kubernetes", "Kubernetes deployment tutorial", "K8s deployment configuration", "How to create Kubernetes deployment?", "Kubernetes pod vs deployment", "Kubernetes deployment strategy", "Getting started Kubernetes deployment"]),
            ("jwt authentication",    ["How does JWT authentication work?", "JWT token authentication explained", "What is JWT auth?", "JSON Web Token authentication", "How to implement JWT auth", "JWT authentication tutorial", "Understanding JWT tokens", "JWT vs session authentication", "Secure JWT implementation", "JWT authentication flow"]),
            ("big o notation",        ["What is Big O notation?", "Big O complexity explained", "How to calculate Big O?", "Algorithm complexity Big O", "Big O notation examples", "Understanding time complexity", "What does O(n) mean?", "Big O space and time complexity", "Algorithm efficiency Big O", "Big O notation tutorial"]),
        ],
        "responses": {
            "async":  "Use 'async def' for coroutines and 'await' to suspend. Run with asyncio.run(). Enables concurrent I/O-bound tasks.",
            "git":    "Merge preserves history with a merge commit. Rebase replays commits for linear history. Use merge for shared branches.",
            "docker": "Containers package apps with dependencies into isolated units. Images are templates; containers are running instances.",
            "react":  "useState returns [state, setState]. Call setState to update and trigger re-render. Never mutate state directly.",
            "sql":    "SQL uses structured schemas with ACID transactions. NoSQL trades consistency for flexibility. Use SQL for relational data.",
            "rest":   "Use HTTP methods (GET/POST/PUT/DELETE), stateless requests, resource URLs. Return proper status codes and use JSON.",
            "python": "[expr for item in iterable if condition]. Faster than loops for simple transforms.",
            "kubernetes": "Deployments manage ReplicaSets ensuring desired pod count. Define with kind: Deployment in YAML.",
            "jwt":    "JWT = header.payload.signature. Server signs with secret key. Client sends in Authorization header.",
            "bigo":   "O(1) constant, O(log n) logarithmic, O(n) linear, O(n²) quadratic. Describes worst-case growth rate.",
        },
    },
    "product_faq": {
        "templates": [
            ("pricing plans",         ["What are your pricing plans?", "How much does it cost?", "Pricing information", "What plans do you offer?", "Cost of subscription", "Pricing tiers explained", "How much is the pro plan?", "Monthly vs annual pricing", "Plan pricing comparison", "What is the cost per month?"]),
            ("free trial",            ["Is there a free trial?", "Can I try it for free?", "Free trial availability", "How long is the free trial?", "Do you offer a free trial?", "Trial period details", "Free tier available?", "Start free trial", "How to get free trial?", "Free trial sign up"]),
            ("data security",         ["How is my data secured?", "Data security practices", "Is my data safe?", "How do you protect user data?", "Data encryption and security", "Security measures for data", "How secure is the platform?", "Data privacy and security", "User data protection policy", "What security do you use?"]),
            ("integrations available",["What integrations do you support?", "Available third-party integrations", "Does it integrate with Slack?", "Supported integrations list", "What tools does it connect with?", "Integration options available", "Can it integrate with our tools?", "List of available integrations", "Supported app integrations", "What does it integrate with?"]),
            ("team collaboration",    ["How does team collaboration work?", "Can multiple users use it?", "Team features available", "Collaborate with my team", "Multi-user account features", "Team workspace setup", "How to add team members?", "Team plan features", "Collaborate on projects together", "Team account management"]),
            ("api access",            ["Do you have an API?", "API access available?", "How to access the API?", "API documentation link", "Can I use the API?", "Programmatic access via API", "REST API available?", "API key and access", "Developer API access", "Getting started with the API"]),
            ("mobile app",            ["Is there a mobile app?", "Mobile application available?", "iOS and Android app", "Download mobile app", "Does it have a mobile version?", "Mobile app download link", "App for smartphone?", "Mobile app features", "Is there an iPhone app?", "Download the app"]),
            ("data export",           ["How do I export my data?", "Data export options", "Can I download my data?", "Export data format", "How to export all data?", "Download my account data", "Data portability options", "Export to CSV or JSON", "How to backup my data?", "Data export and download"]),
            ("uptime sla",            ["What is your uptime guarantee?", "SLA and uptime commitment", "Service level agreement details", "How reliable is the service?", "Uptime percentage guarantee", "SLA terms and conditions", "Service availability guarantee", "Reliability and uptime SLA", "What uptime do you guarantee?", "SLA for enterprise customers"]),
            ("gdpr compliance",       ["Are you GDPR compliant?", "GDPR compliance status", "How do you handle GDPR?", "Data privacy GDPR compliance", "GDPR and data protection", "Is the product GDPR ready?", "GDPR compliance documentation", "Privacy regulations compliance", "EU data protection compliance", "GDPR data processing agreement"]),
        ],
        "responses": {
            "pricing":      "Starter ($29/mo), Growth ($99/mo), Enterprise (custom). Annual saves 20%.",
            "trial":        "14-day free trial, no credit card required. All Pro features included.",
            "security":     "AES-256 at rest, TLS 1.3 in transit, SOC 2 Type II certified, GDPR compliant.",
            "integrations": "50+ integrations: Slack, Jira, GitHub, Salesforce, HubSpot, Zapier and more.",
            "team":         "Unlimited members on team plans. Admins manage roles, permissions, workspaces.",
            "api":          "Full REST API available. 1,000 req/min on Growth, unlimited on Enterprise.",
            "mobile":       "iOS and Android apps with full feature parity and offline mode.",
            "export":       "Export as CSV, JSON, or PDF from Settings > Data Export. Ready within 24hrs.",
            "uptime":       "99.9% SLA for Growth, 99.99% for Enterprise. Credits for downtime.",
            "gdpr":         "Fully GDPR compliant. DPA available. Data in EU. Erasure requests in 30 days.",
        },
    },
    "medical_information": {
        "templates": [
            ("high blood pressure",   ["What is high blood pressure?", "Hypertension explained", "High blood pressure symptoms", "What causes high blood pressure?", "Hypertension treatment options", "How to lower blood pressure?", "Blood pressure normal range", "High BP risk factors", "Managing hypertension", "High blood pressure complications"]),
            ("type 2 diabetes",       ["What is type 2 diabetes?", "Type 2 diabetes explained", "Symptoms of type 2 diabetes", "How is type 2 diabetes treated?", "Type 2 diabetes management", "What causes type 2 diabetes?", "Diabetes type 2 risk factors", "Managing blood sugar diabetes", "Type 2 diabetes diet", "Insulin resistance diabetes"]),
            ("common cold treatment", ["How do you treat a common cold?", "Common cold remedies", "Cold symptoms treatment", "How long does a cold last?", "Best treatment for cold", "Cold vs flu differences", "How to recover from cold faster", "Treating cold symptoms at home", "Cold medicine and remedies", "Common cold duration and treatment"]),
            ("covid vaccine",         ["How do COVID vaccines work?", "COVID-19 vaccine mechanism", "mRNA vaccine explained", "COVID vaccine side effects", "Are COVID vaccines safe?", "COVID vaccination benefits", "How effective is COVID vaccine?", "COVID booster vaccine info", "COVID vaccine types comparison", "COVID vaccine immune response"]),
            ("mental health anxiety", ["What are symptoms of anxiety?", "Anxiety disorder symptoms", "How to manage anxiety?", "Anxiety treatment options", "Dealing with anxiety", "Anxiety vs normal worry", "Types of anxiety disorders", "Anxiety medication options", "Therapy for anxiety", "Anxiety self-help techniques"]),
            ("vitamin d deficiency",  ["What are symptoms of vitamin D deficiency?", "Vitamin D deficiency signs", "How to treat vitamin D deficiency?", "Low vitamin D symptoms", "Vitamin D deficiency causes", "Vitamin D supplement dosage", "Vitamin D and bone health", "How much vitamin D do I need?", "Vitamin D deficiency treatment", "Sun exposure and vitamin D"]),
            ("migraine headache",     ["What causes migraines?", "Migraine headache symptoms", "How to treat a migraine?", "Migraine triggers to avoid", "Migraine vs tension headache", "Migraine treatment options", "Preventing migraine attacks", "Migraine medication list", "How long does migraine last?", "Chronic migraine management"]),
            ("sleep disorders",       ["What are common sleep disorders?", "Types of sleep disorders", "Insomnia causes and treatment", "How to treat sleep problems?", "Sleep disorder symptoms", "Sleep apnea explained", "Improving sleep quality", "Sleep disorder diagnosis", "Treatment for insomnia", "Sleep hygiene tips"]),
            ("back pain causes",      ["What causes lower back pain?", "Lower back pain causes", "Back pain treatment options", "How to relieve back pain?", "Chronic back pain causes", "Back pain exercises", "Lower back pain remedies", "When to see doctor for back pain?", "Back pain relief at home", "Preventing lower back pain"]),
            ("antibiotic usage",      ["When should I take antibiotics?", "Antibiotic use guidelines", "How do antibiotics work?", "Antibiotic resistance explained", "Correct antibiotic usage", "Side effects of antibiotics", "Completing antibiotic course", "Antibiotic vs antiviral", "When are antibiotics needed?", "Antibiotic treatment duration"]),
        ],
        "responses": {
            "blood":    "Normal BP below 120/80. High BP (130+/80+) treated with lifestyle changes and medication.",
            "diabetes": "Type 2 impairs insulin use. Managed via diet, exercise, metformin, and HbA1c monitoring.",
            "cold":     "No cure. Treat symptoms with rest, fluids, decongestants. Lasts 7-10 days.",
            "covid":    "mRNA vaccines trigger immune response. 90-95% effective. Side effects last 1-2 days.",
            "anxiety":  "Treated with CBT therapy, SSRIs. Affects 18% of adults. Causes excessive worry.",
            "vitamin":  "Treat with D3 supplements (1000-4000 IU/day) and sun exposure.",
            "migraine": "Treated with triptans, NSAIDs. Triggers: stress, hormones, certain foods.",
            "sleep":    "Disorders: insomnia, sleep apnea. Treat with CBT-I, CPAP, sleep hygiene.",
            "back":     "Causes: muscle strain, disc herniation. Treat with rest, NSAIDs, physio.",
            "antibiotic":"Complete full course. Only for bacterial infections. Overuse causes resistance.",
        },
    },
    "general_knowledge": {
        "templates": [
            ("what is ai",            ["What is artificial intelligence?", "Explain artificial intelligence", "AI definition and overview", "What does AI mean?", "Artificial intelligence explained", "How does AI work?", "Introduction to AI", "What can AI do?", "AI basics explained", "Overview of artificial intelligence"]),
            ("climate change",        ["What is climate change?", "Explain climate change", "What causes climate change?", "Climate change effects", "Global warming explained", "Climate change impact", "What is global warming?", "Causes of climate change", "Climate change overview", "Effects of global warming"]),
            ("blockchain technology", ["What is blockchain?", "Blockchain technology explained", "How does blockchain work?", "What is a blockchain?", "Blockchain overview", "Blockchain use cases", "Explain blockchain technology", "What is distributed ledger?", "Blockchain basics", "How blockchain works simply"]),
            ("how internet works",    ["How does the internet work?", "Explain how the internet works", "What is the internet?", "Internet infrastructure explained", "How data travels on internet", "Internet protocols explained", "How websites work", "How does the web work?", "Internet basics explained", "TCP IP explained simply"]),
            ("quantum computing",     ["What is quantum computing?", "Quantum computing explained", "How does quantum computing work?", "Quantum vs classical computing", "Quantum computer basics", "What can quantum computers do?", "Explain quantum computing simply", "Quantum computing overview", "Future of quantum computing", "Quantum bits explained"]),
            ("renewable energy",      ["What is renewable energy?", "Types of renewable energy", "Explain renewable energy sources", "Solar and wind energy", "Renewable vs fossil fuels", "Benefits of renewable energy", "How solar energy works", "Renewable energy overview", "Clean energy sources explained", "Future of renewable energy"]),
            ("machine learning",      ["What is machine learning?", "Machine learning explained", "How does machine learning work?", "ML basics for beginners", "Introduction to machine learning", "What can machine learning do?", "Machine learning overview", "AI vs machine learning", "Getting started machine learning", "Machine learning definition"]),
            ("cryptocurrency bitcoin",["What is Bitcoin?", "Bitcoin explained simply", "How does Bitcoin work?", "What is cryptocurrency?", "Bitcoin vs traditional currency", "How to buy Bitcoin?", "Bitcoin blockchain explained", "Cryptocurrency basics", "What is digital currency?", "Bitcoin investment overview"]),
            ("dna genetics",          ["What is DNA?", "DNA explained simply", "How does DNA work?", "What is genetics?", "DNA and heredity", "Genes and DNA explained", "How genes work", "DNA structure and function", "Genetics basics", "What is a gene?"]),
            ("space exploration",     ["How do rockets work?", "Space exploration explained", "How do we explore space?", "Rocket propulsion basics", "How does a rocket engine work?", "Space mission overview", "How astronauts travel to space", "Rocket science basics", "Space shuttle how it works", "Getting to space explained"]),
        ],
        "responses": {
            "ai":         "AI enables machines to perform tasks requiring human intelligence: learning, reasoning, perception.",
            "climate":    "Long-term shifts in global temperatures caused by burning fossil fuels and greenhouse gases.",
            "blockchain": "Distributed ledger where records are linked cryptographically. Powers cryptocurrencies.",
            "internet":   "Global network via TCP/IP. Data travels in packets through routers. DNS maps domains to IPs.",
            "quantum":    "Qubits exist in superposition (0, 1, or both). Useful for cryptography and optimisation.",
            "renewable":  "Sources: solar, wind, hydro, geothermal. No emissions, naturally replenished.",
            "ml":         "Computers learn from data without explicit programming. Types: supervised, unsupervised, RL.",
            "bitcoin":    "Decentralised digital currency using blockchain. Limited to 21 million coins.",
            "dna":        "Double helix of nucleotides (ACGT). Genes encode proteins. Inherited from both parents.",
            "space":      "Newton's third law: expelled exhaust creates thrust. Escape velocity = 11.2 km/s.",
        },
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CORPUS BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_corpus(n_test: int = 5000) -> dict:
    """Returns {domain: [{query, response, group, is_warmup}]}"""
    corpus      = {}
    n_per_domain= n_test // len(DOMAINS)
    prefixes    = ["", "Please tell me ", "Can you explain ", "I need to know ",
                   "Quick question: ", "Help me understand ", "Could you tell me ",
                   "I was wondering ", ""]
    suffixes    = ["", "?", " please", " - need help", " asap", " thanks", ""]

    for domain, cfg in DOMAINS.items():
        templates = cfg["templates"]
        responses = cfg["responses"]
        queries   = []

        for group_key, base_queries in templates:
            resp_key = next((k for k in responses if k in group_key),
                            list(responses.keys())[0])
            response = responses[resp_key]
            expanded = list(base_queries)
            for _ in range(190):
                base    = random.choice(base_queries)
                variant = (random.choice(prefixes) +
                           base.rstrip("?") +
                           random.choice(suffixes)).strip()
                if not variant.endswith(("?", ".")):
                    variant += "?"
                expanded.append(variant)
            random.shuffle(expanded)
            for i, q in enumerate(expanded[:200]):
                queries.append({
                    "query":     q,
                    "response":  response,
                    "group":     group_key,
                    "domain":    domain,
                    "is_warmup": i < 100,
                })

        random.shuffle(queries)
        corpus[domain] = queries[:n_per_domain * 2]

    return corpus


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESULT DATA CLASS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    query:         str
    domain:        str
    group:         str
    is_warmup:     bool
    cache_hit:     bool
    similarity:    float
    matched_group: str
    latency_ms:    float
    correct:       bool


# ══════════════════════════════════════════════════════════════════════════════
# 6.  BENCHMARK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run(corpus: dict, threshold: float, use_sulci: bool, verbose: bool = True) -> list:
    if use_sulci:
        db_path = os.path.join(args.out, "sulci_bench_db")
        cache   = _SulciWrapper(threshold, db_path)
        engine  = "sulci.Cache (SQLite + MiniLM)"
    else:
        cache  = _BuiltinCache(threshold)
        engine = "built-in TF-IDF engine"

    all_items  = []
    for items in corpus.values():
        all_items.extend(items)

    warmup = [x for x in all_items if x["is_warmup"]]
    test   = [x for x in all_items if not x["is_warmup"]]

    if verbose:
        print(f"\n{'='*58}")
        print(f"  Sulci Benchmark  |  threshold={threshold}")
        print(f"  Engine: {engine}")
        print(f"{'='*58}")
        print(f"  Warmup : {len(warmup):,}  |  Test : {len(test):,}")
        print(f"{'='*58}\n")

    results = []

    for item in warmup:
        t0 = time.perf_counter()
        cache.set(item["query"], item["response"],
                  group=item["group"], domain=item["domain"])
        ms = (time.perf_counter() - t0) * 1000
        results.append(Result(
            query=item["query"], domain=item["domain"], group=item["group"],
            is_warmup=True, cache_hit=False, similarity=1.0,
            matched_group="", latency_ms=round(ms, 3), correct=True,
        ))

    for i, item in enumerate(test):
        t0  = time.perf_counter()
        resp, sim, matched = cache.get(item["query"])
        ms  = (time.perf_counter() - t0) * 1000

        if resp is None:
            cache.set(item["query"], item["response"],
                      group=item["group"], domain=item["domain"])
            results.append(Result(
                query=item["query"], domain=item["domain"], group=item["group"],
                is_warmup=False, cache_hit=False, similarity=sim,
                matched_group="", latency_ms=round(ms, 3), correct=True,
            ))
        else:
            m_group = getattr(matched, "group", "") if matched else ""
            correct = (m_group == item["group"])
            results.append(Result(
                query=item["query"], domain=item["domain"], group=item["group"],
                is_warmup=False, cache_hit=True, similarity=sim,
                matched_group=m_group, latency_ms=round(ms, 3), correct=correct,
            ))

        if verbose and (i + 1) % 500 == 0:
            done  = [r for r in results if not r.is_warmup]
            hits  = sum(1 for r in done if r.cache_hit)
            print(f"  [{i+1:5,}/{len(test):,}]  "
                  f"hit rate: {hits/len(done):.1%}  "
                  f"entries: {len(warmup) + i + 1:,}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════

def percentile(lst, p):
    if not lst: return 0.0
    s = sorted(lst)
    return s[int(len(s) * p / 100)]


def summary(results: list, threshold: float) -> dict:
    test = [r for r in results if not r.is_warmup]
    hits = [r for r in test if r.cache_hit]
    miss = [r for r in test if not r.cache_hit]
    fps  = [r for r in hits if not r.correct]
    COST = 0.005

    return {
        "threshold":             threshold,
        "total_queries":         len(test),
        "cache_hits":            len(hits),
        "cache_misses":          len(miss),
        "hit_rate":              round(len(hits) / len(test), 4) if test else 0,
        "false_positives":       len(fps),
        "false_positive_rate":   round(len(fps) / len(hits), 4) if hits else 0,
        "avg_similarity_hits":   round(sum(r.similarity for r in hits) / len(hits), 4) if hits else 0,
        "latency_hit_p50_ms":    round(percentile([r.latency_ms for r in hits], 50), 3),
        "latency_hit_p95_ms":    round(percentile([r.latency_ms for r in hits], 95), 3),
        "latency_miss_p50_ms":   round(percentile([r.latency_ms for r in miss], 50), 3),
        "latency_miss_p95_ms":   round(percentile([r.latency_ms for r in miss], 95), 3),
        "baseline_cost_usd":     round(len(test) * COST, 4),
        "actual_cost_usd":       round(len(miss) * COST, 4),
        "saved_cost_usd":        round(len(hits) * COST, 4),
        "cost_reduction_pct":    round(len(hits) / len(test) * 100, 2) if test else 0,
    }


def domain_breakdown(results: list) -> list:
    test = [r for r in results if not r.is_warmup]
    rows = []
    COST = 0.005
    for domain in DOMAINS:
        d   = [r for r in test if r.domain == domain]
        h   = [r for r in d if r.cache_hit]
        m   = [r for r in d if not r.cache_hit]
        fp  = [r for r in h if not r.correct]
        rows.append({
            "domain":             domain,
            "total":              len(d),
            "hits":               len(h),
            "misses":             len(m),
            "hit_rate_pct":       round(len(h)/len(d)*100, 1) if d else 0,
            "false_positives":    len(fp),
            "fp_rate_pct":        round(len(fp)/len(h)*100, 2) if h else 0,
            "avg_sim_hits":       round(sum(r.similarity for r in h)/len(h), 4) if h else 0,
            "saved_usd":          round(len(h)*COST, 3),
            "cost_reduction_pct": round(len(h)/len(d)*100, 1) if d else 0,
        })
    return rows


def time_series(results: list, window: int = 100) -> list:
    test = [r for r in results if not r.is_warmup]
    rows = []
    for i in range(0, len(test), window):
        chunk    = test[i:i+window]
        hits     = sum(1 for r in chunk if r.cache_hit)
        cum      = test[:i+len(chunk)]
        cum_hits = sum(1 for r in cum if r.cache_hit)
        rows.append({
            "batch":                  i // window + 1,
            "queries_processed":      i + len(chunk),
            "window_hit_rate_pct":    round(hits/len(chunk)*100, 1) if chunk else 0,
            "cumulative_hit_rate_pct":round(cum_hits/len(cum)*100, 1) if cum else 0,
        })
    return rows


def false_positives_report(results: list) -> list:
    fps = [r for r in results if not r.is_warmup and r.cache_hit and not r.correct]
    return sorted([{
        "domain":        r.domain,
        "group":         r.group,
        "matched_group": r.matched_group,
        "similarity":    r.similarity,
        "query":         r.query[:100],
    } for r in fps[:100]], key=lambda x: -x["similarity"])


# ══════════════════════════════════════════════════════════════════════════════
# 8.  I/O
# ══════════════════════════════════════════════════════════════════════════════

def save_json(obj, name):
    path = os.path.join(args.out, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved {path}")


def save_csv(rows, name):
    if not rows: return
    path = os.path.join(args.out, name)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  THRESHOLD SWEEP
# ══════════════════════════════════════════════════════════════════════════════

def sweep(corpus: dict, use_sulci: bool) -> list:
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.95]
    rows = []
    print("\n── Threshold sweep ──────────────────────────────────────")
    for t in thresholds:
        res = run(corpus, threshold=t, use_sulci=use_sulci, verbose=False)
        s   = summary(res, t)
        print(f"  t={t:.2f}  hit={s['hit_rate']:.1%}  "
              f"fp={s['false_positive_rate']:.2%}  "
              f"saved={s['cost_reduction_pct']:.1f}%")
        rows.append({
            "threshold":          t,
            "hit_rate_pct":       round(s["hit_rate"]*100, 1),
            "false_positive_pct": round(s["false_positive_rate"]*100, 2),
            "cost_reduction_pct": s["cost_reduction_pct"],
            "hits":               s["cache_hits"],
            "misses":             s["cache_misses"],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("\n◈ Sulci Benchmark")
    print(f"  Building {args.queries:,}-query corpus...")
    corpus = build_corpus(n_test=args.queries)
    print(f"  Done ({sum(len(v) for v in corpus.values()):,} total queries)\n")

    results = run(corpus, args.threshold, args.use_sulci, verbose=True)

    print("\n── Saving results ───────────────────────────────────────")
    s = summary(results, args.threshold)
    save_json(s,                          "summary.json")
    save_csv(domain_breakdown(results),   "domain_breakdown.csv")
    save_csv(time_series(results),        "time_series.csv")
    save_csv(false_positives_report(results), "false_positives.csv")

    if not args.no_sweep:
        sw = sweep(corpus, args.use_sulci)
        save_csv(sw, "threshold_sweep.csv")

    elapsed = time.time() - t0
    print(f"\n{'='*58}")
    print(f"  RESULTS  |  threshold={args.threshold}")
    print(f"{'='*58}")
    print(f"  Queries        : {s['total_queries']:,}")
    print(f"  Hits           : {s['cache_hits']:,}  ({s['hit_rate']:.1%})")
    print(f"  False positives: {s['false_positives']} ({s['false_positive_rate']:.2%})")
    print(f"  Latency (hit)  : {s['latency_hit_p50_ms']:.2f}ms p50  /  {s['latency_hit_p95_ms']:.2f}ms p95")
    print(f"  Latency (miss) : {s['latency_miss_p50_ms']:.2f}ms p50  /  {s['latency_miss_p95_ms']:.2f}ms p95")
    print(f"  Cost saved     : ${s['saved_cost_usd']:.2f}  ({s['cost_reduction_pct']:.1f}%)")
    print(f"  Completed in   : {elapsed:.1f}s")
    print(f"  Results in     : {args.out}/")
    print(f"{'='*58}\n")

    print("  Domain breakdown:")
    for row in domain_breakdown(results):
        print(f"    {row['domain']:22s}  hit={row['hit_rate_pct']:5.1f}%  "
              f"fp={row['fp_rate_pct']:4.1f}%  saved=${row['saved_usd']:.2f}")
    print()


if __name__ == "__main__":
    main()
