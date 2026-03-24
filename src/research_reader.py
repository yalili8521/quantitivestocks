#!/usr/bin/env python3
"""
Research Reader — Ingests daily markdown research reports into ETF screening.
===============================================================================

Reads markdown files from research/daily/YYYY-MM-DD.md, extracts:
  - Ticker mentions (with sentiment: bullish/bearish/neutral)
  - Sector/category signals (e.g. "semiconductors strong", "energy weak")
  - Macro regime signals (risk-on/risk-off, rate expectations)
  - Thematic keywords (AI, reshoring, commodities supercycle, etc.)

These signals become bias weights in the weekly ETF screening composite score.

Markdown format (flexible — parser is forgiving):
  ## Macro
  Risk-on environment, soft landing narrative intact.
  Fed expected to hold rates.

  ## Sectors
  - Semiconductors: STRONG — AI capex cycle accelerating
  - Energy: WEAK — demand concerns from China slowdown
  - Gold/precious metals: BULLISH — real yields falling

  ## Tickers
  - SMH: bullish, AI infrastructure spending
  - XLE: bearish, OPEC+ uncertainty
  - GLD: bullish, central bank buying

  ## Themes
  AI infrastructure, reshoring, commodity supercycle, EM recovery

Usage:
    from research_reader import ResearchReader
    reader = ResearchReader()
    signals = reader.read_last_n_days(7)
    bias = reader.compute_bias_weights(signals)
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Set

log = logging.getLogger("research_reader")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH_DIR = os.path.join(PROJECT_ROOT, "research", "daily")

# Sentiment keywords → numeric score
BULLISH_WORDS = {
    "bullish", "strong", "buy", "long", "positive", "outperform",
    "upgrade", "accumulate", "breakout", "rally", "upside",
    "overweight", "momentum", "accelerating", "recovery",
}
BEARISH_WORDS = {
    "bearish", "weak", "sell", "short", "negative", "underperform",
    "downgrade", "avoid", "breakdown", "decline", "downside",
    "underweight", "deteriorating", "slowdown", "risk-off",
}

# Map sector keywords to ETF categories (matches etf_screener.py categories)
SECTOR_CATEGORY_MAP = {
    # Technology / semis
    "semiconductor": "us_sector", "semis": "us_sector", "chips": "us_sector",
    "tech": "us_sector", "software": "us_sector", "ai": "us_sector",
    "cybersecurity": "us_sector", "cloud": "us_sector",
    # Energy
    "energy": "us_sector", "oil": "commodities", "natural gas": "commodities",
    # Commodities / precious metals
    "gold": "commodities", "silver": "commodities", "precious metals": "commodities",
    "copper": "commodities", "uranium": "commodities", "mining": "commodities",
    "commodities": "commodities", "commodity": "commodities",
    # Financials
    "financials": "us_sector", "banks": "us_sector", "banking": "us_sector",
    # Healthcare
    "healthcare": "us_sector", "biotech": "us_sector", "pharma": "us_sector",
    # International
    "emerging markets": "intl_em", "em": "intl_em", "china": "intl_em",
    "taiwan": "intl_em", "brazil": "intl_em", "india": "intl_em",
    "europe": "intl_developed", "japan": "intl_developed",
    # Fixed income
    "bonds": "fixed_income", "treasuries": "fixed_income", "rates": "fixed_income",
    "fixed income": "fixed_income", "credit": "fixed_income",
    # Crypto
    "crypto": "crypto_etf", "bitcoin": "crypto_etf",
    # Factors
    "value": "us_factor", "growth": "us_factor", "small cap": "us_factor",
    "momentum": "us_factor",
}

# Known ticker → category override (for tickers not in etf_screener seed)
TICKER_CATEGORY_HINTS = {
    # Tech / semis
    "SMH": "us_sector", "SOXX": "us_sector", "XLK": "us_sector",
    "IGV": "us_sector", "BOTZ": "us_sector", "CIBR": "us_sector",
    "SKYY": "us_sector", "ARKK": "us_sector",
    # Sector SPDRs
    "XLE": "us_sector", "XLF": "us_sector", "XLV": "us_sector",
    "XLI": "us_sector", "XLP": "us_sector", "XLY": "us_sector",
    "XLU": "us_sector", "XLB": "us_sector", "XLRE": "us_sector",
    # Energy
    "OIH": "us_sector", "AMLP": "us_sector", "UNG": "commodities",
    # Infrastructure / industrials
    "PAVE": "us_sector", "IGF": "us_sector", "ITA": "us_sector",
    "ITB": "us_sector", "XHB": "us_sector",
    # Commodities
    "GLD": "commodities", "SLV": "commodities", "GDX": "commodities",
    "GDXJ": "commodities", "USO": "commodities", "CPER": "commodities",
    "URNM": "commodities", "IAU": "commodities", "PDBC": "commodities",
    "DBA": "commodities",
    # International developed
    "EWJ": "intl_developed", "VGK": "intl_developed", "EWG": "intl_developed",
    "EWA": "intl_developed", "EWU": "intl_developed", "EWC": "intl_developed",
    # International EM
    "EEM": "intl_em", "EWT": "intl_em", "MCHI": "intl_em",
    "EWZ": "intl_em", "INDA": "intl_em", "VWO": "intl_em",
    "KWEB": "intl_em",
    # Fixed income
    "TLT": "fixed_income", "HYG": "fixed_income", "LQD": "fixed_income",
    "IEF": "fixed_income", "EMB": "fixed_income", "SHY": "fixed_income",
    "BND": "fixed_income", "TIP": "fixed_income",
    # Crypto ETFs
    "IBIT": "crypto_etf", "FBTC": "crypto_etf", "GBTC": "crypto_etf",
    "ETHA": "crypto_etf",
    # Factors / broad market
    "SPY": "us_factor", "QQQ": "us_factor", "IWM": "us_factor",
    "SLYV": "us_factor", "VTV": "us_factor", "MTUM": "us_factor",
    "QUAL": "us_factor", "IWF": "us_factor",
}


@dataclass
class TickerMention:
    """A single ticker mention extracted from research."""
    symbol: str
    sentiment: float       # -1.0 (bearish) to +1.0 (bullish)
    context: str           # surrounding text
    date: str              # YYYY-MM-DD


@dataclass
class SectorSignal:
    """A sector/category-level signal."""
    category: str          # matches etf_screener categories
    sentiment: float       # -1.0 to +1.0
    keywords: List[str]    # triggering keywords
    date: str


@dataclass
class MacroSignal:
    """Macro regime signal."""
    regime: str            # "risk_on", "risk_off", "neutral"
    rate_direction: str    # "hawkish", "dovish", "neutral"
    confidence: float      # 0-1
    date: str


@dataclass
class ResearchSignals:
    """Aggregated signals from one or more research reports."""
    ticker_mentions: List[TickerMention] = field(default_factory=list)
    sector_signals: List[SectorSignal] = field(default_factory=list)
    macro_signals: List[MacroSignal] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    dates_read: List[str] = field(default_factory=list)


class ResearchReader:
    """Reads and parses daily markdown research reports."""

    def __init__(self, research_dir: str = RESEARCH_DIR):
        self.research_dir = research_dir

    def _list_available_dates(self) -> List[str]:
        """List available research file dates, sorted descending."""
        if not os.path.isdir(self.research_dir):
            return []
        dates = []
        for fname in os.listdir(self.research_dir):
            if fname.endswith(".md"):
                date_str = fname.replace(".md", "")
                # Validate date format
                try:
                    datetime.strptime(date_str, "%Y-%m-%d")
                    dates.append(date_str)
                except ValueError:
                    continue
        return sorted(dates, reverse=True)

    def read_last_n_days(self, n: int = 7) -> ResearchSignals:
        """Read and parse the last N days of research reports."""
        available = self._list_available_dates()
        if not available:
            log.info("No research files found in %s", self.research_dir)
            return ResearchSignals()

        # Take at most N most recent files
        to_read = available[:n]
        log.info("Reading %d research files: %s", len(to_read),
                 ", ".join(to_read))

        all_signals = ResearchSignals(dates_read=to_read)
        for date_str in to_read:
            path = os.path.join(self.research_dir, f"{date_str}.md")
            try:
                with open(path, encoding="utf-8") as f:
                    content = f.read()
                self._parse_report(content, date_str, all_signals)
            except Exception as exc:
                log.warning("Failed to read %s: %s", path, exc)

        log.info("Extracted: %d ticker mentions, %d sector signals, %d macro signals, %d themes",
                 len(all_signals.ticker_mentions), len(all_signals.sector_signals),
                 len(all_signals.macro_signals), len(all_signals.themes))
        return all_signals

    def _classify_heading(self, heading: str) -> str:
        """Classify a section heading into a category using fuzzy matching.

        Real-world reports use varied heading styles like:
          "1. Macro & Geopolitical Events", "ETF Spotlight",
          "Technology & Hot Topics", "Energy Sector", etc.
        """
        h = heading.lower()
        # Strip leading numbers/punctuation: "1. Macro" → "macro"
        h = re.sub(r"^[\d\.\)\-\s]+", "", h).strip()

        # Macro / regime
        macro_kw = ("macro", "geopolit", "regime", "market overview",
                    "market environment", "overview")
        if any(k in h for k in macro_kw):
            return "macro"

        # Explicit ticker/ETF sections
        ticker_kw = ("etf spotlight", "etf play", "etf pick", "ticker",
                     "symbol", "picks", "positions", "highest-conviction",
                     "conviction signal", "top trade")
        if any(k in h for k in ticker_kw):
            return "tickers"

        # Sector-specific headings
        sector_kw = ("sector", "energy", "tech", "infrastructure",
                     "commodit", "financ", "healthcare", "crypto",
                     "digital asset", "under-the-radar", "opportunity",
                     "small-cap", "rotation")
        if any(k in h for k in sector_kw):
            return "sector_and_tickers"

        # Themes
        theme_kw = ("theme", "narrative", "idea")
        if any(k in h for k in theme_kw):
            return "themes"

        return "general"

    # Signal Manifest — explicit sentiment vocabulary
    MANIFEST_SECTOR_SCORES = {
        "STRONG": 1.0, "BULLISH": 0.7, "NEUTRAL": 0.0,
        "WEAK": -0.7, "BEARISH": -1.0,
    }
    MANIFEST_TICKER_SCORES = {
        "bullish": 0.8, "leaning bullish": 0.4, "neutral": 0.0,
        "leaning bearish": -0.4, "bearish": -0.8,
    }

    def _parse_report(self, content: str, date_str: str,
                      signals: ResearchSignals) -> None:
        """Parse a single markdown report into signals.

        If a Signal Manifest section exists, its structured data is used
        preferentially for macro, sectors, tickers, and themes. The narrative
        sections are still scanned for bold-ticker extraction to catch any
        tickers not listed in the manifest.
        """
        sections = self._split_sections(content)

        # Check for Signal Manifest — if present, use it as primary source
        manifest_found = self._try_parse_signal_manifest(
            content, date_str, signals
        )

        if manifest_found:
            log.info("[%s] Signal Manifest found — using structured data", date_str)
            # Still extract bold tickers from narrative sections for coverage
            for heading, body in sections.items():
                category = self._classify_heading(heading)
                if category != "macro" and heading != "Signal Manifest":
                    self._extract_bold_tickers(body, date_str, signals)
        else:
            # Fallback: narrative parsing (original behavior)
            for heading, body in sections.items():
                category = self._classify_heading(heading)
                if category == "macro":
                    self._parse_macro_section(body, date_str, signals)

            for heading, body in sections.items():
                category = self._classify_heading(heading)
                if category == "tickers":
                    self._parse_ticker_section(body, date_str, signals)
                    self._extract_bold_tickers(body, date_str, signals)
                elif category == "sector_and_tickers":
                    self._parse_sector_section(body, date_str, signals)
                    self._extract_bold_tickers(body, date_str, signals)
                elif category == "themes":
                    self._parse_themes_section(body, signals)
                elif category == "general":
                    self._extract_bold_tickers(body, date_str, signals)

        # Deduplicate: if a ticker was mentioned multiple times, keep highest-abs sentiment
        self._deduplicate_mentions(signals, date_str)

    def _try_parse_signal_manifest(self, content: str, date_str: str,
                                    signals: ResearchSignals) -> bool:
        """Parse the Signal Manifest section if present.

        The Signal Manifest uses explicit vocabulary:
          Sectors: STRONG / BULLISH / NEUTRAL / WEAK / BEARISH
          Tickers: bullish / leaning bullish / neutral / leaning bearish / bearish

        Returns True if a manifest was found and parsed.
        """
        # Find the Signal Manifest section
        manifest_match = re.search(
            r"^##\s+Signal\s+Manifest\b.*?\n(.*)",
            content, re.MULTILINE | re.DOTALL | re.IGNORECASE,
        )
        if not manifest_match:
            return False

        manifest_body = manifest_match.group(1)

        # Split manifest into sub-sections (## Macro, ## Sectors, etc.)
        sub_sections: Dict[str, str] = {}
        current_key = "_preamble"
        current_lines: List[str] = []
        for line in manifest_body.splitlines():
            m = re.match(r"^#{1,3}\s+(.+)", line)
            if m:
                if current_lines:
                    sub_sections[current_key] = "\n".join(current_lines)
                current_key = m.group(1).strip().lower()
                current_lines = []
            else:
                current_lines.append(line)
        if current_lines:
            sub_sections[current_key] = "\n".join(current_lines)

        parsed_any = False

        # --- Macro ---
        if "macro" in sub_sections:
            self._parse_manifest_macro(sub_sections["macro"], date_str, signals)
            parsed_any = True

        # --- Sectors ---
        if "sectors" in sub_sections:
            self._parse_manifest_sectors(sub_sections["sectors"], date_str, signals)
            parsed_any = True

        # --- Tickers ---
        if "tickers" in sub_sections:
            self._parse_manifest_tickers(sub_sections["tickers"], date_str, signals)
            parsed_any = True

        # --- Themes ---
        if "themes" in sub_sections:
            self._parse_themes_section(sub_sections["themes"], signals)
            parsed_any = True

        return parsed_any

    def _parse_manifest_macro(self, body: str, date_str: str,
                               signals: ResearchSignals) -> None:
        """Parse macro from Signal Manifest with explicit regime labels.

        Looks for explicit regime keywords: STAGFLATION, RISK-ON, RISK-OFF,
        NEUTRAL, EXPANSION, CONTRACTION, etc.
        """
        body_lower = body.lower()

        # Explicit regime labels (case-insensitive)
        regime_map = {
            "stagflation": "risk_off",
            "risk-off": "risk_off",
            "risk off": "risk_off",
            "contraction": "risk_off",
            "recession": "risk_off",
            "risk-on": "risk_on",
            "risk on": "risk_on",
            "expansion": "risk_on",
            "goldilocks": "risk_on",
            "neutral": "neutral",
        }
        regime = "neutral"
        for label, mapped in regime_map.items():
            if label in body_lower:
                regime = mapped
                break

        # Rate direction — look for explicit statements
        rate_dir = "neutral"
        if any(k in body_lower for k in ("dovish", "easing", "yields eased",
                                          "yields declined", "rate cut")):
            rate_dir = "dovish"
        elif any(k in body_lower for k in ("hawkish", "tightening", "rate hike",
                                            "higher for longer")):
            rate_dir = "hawkish"

        # Higher confidence for manifest (structured data)
        confidence = 0.9

        signals.macro_signals.append(
            MacroSignal(regime=regime, rate_direction=rate_dir,
                        confidence=confidence, date=date_str)
        )

    def _parse_manifest_sectors(self, body: str, date_str: str,
                                 signals: ResearchSignals) -> None:
        """Parse sector lines from Signal Manifest.

        Expected format: - Category: STRONG/BULLISH/NEUTRAL/WEAK/BEARISH — description
        """
        sector_re = re.compile(
            r"[-*]\s*(.+?):\s*(STRONG|BULLISH|NEUTRAL|WEAK|BEARISH)\b\s*(?:—\s*(.*))?",
            re.IGNORECASE,
        )
        for line in body.splitlines():
            m = sector_re.match(line.strip())
            if not m:
                continue
            sector_name = m.group(1).strip().lower()
            call = m.group(2).upper()
            sentiment = self.MANIFEST_SECTOR_SCORES.get(call, 0.0)

            # Map to categories using word-boundary matching to avoid
            # false positives like "em" inside "semiconductors"
            matched_cats: Set[str] = set()
            matched_kw: List[str] = []
            for keyword, category in SECTOR_CATEGORY_MAP.items():
                # Use word boundary for short keywords (<=3 chars)
                if len(keyword) <= 3:
                    if re.search(r"\b" + re.escape(keyword) + r"\b", sector_name):
                        matched_cats.add(category)
                        matched_kw.append(keyword)
                else:
                    if keyword in sector_name:
                        matched_cats.add(category)
                        matched_kw.append(keyword)

            # If no category matched, treat as general sector
            if not matched_cats:
                matched_cats.add("us_sector")
                matched_kw.append(sector_name)

            for cat in matched_cats:
                signals.sector_signals.append(
                    SectorSignal(category=cat, sentiment=sentiment,
                                 keywords=matched_kw, date=date_str)
                )

    def _parse_manifest_tickers(self, body: str, date_str: str,
                                 signals: ResearchSignals) -> None:
        """Parse ticker lines from Signal Manifest.

        Expected format: - TICKER: bullish/leaning bullish/neutral/leaning bearish/bearish, description
        """
        ticker_re = re.compile(
            r"[-*]\s*([A-Z]{2,5})\s*:\s*"
            r"(leaning bullish|leaning bearish|bullish|bearish|neutral)"
            r"\s*[,—\-]?\s*(.*)",
            re.IGNORECASE,
        )
        for line in body.splitlines():
            m = ticker_re.match(line.strip())
            if not m:
                continue
            symbol = m.group(1).upper()
            call = m.group(2).lower()
            context = m.group(3).strip() if m.group(3) else ""
            sentiment = self.MANIFEST_TICKER_SCORES.get(call, 0.0)

            signals.ticker_mentions.append(
                TickerMention(symbol=symbol, sentiment=sentiment,
                              context=context[:120], date=date_str)
            )

    def _split_sections(self, content: str) -> Dict[str, str]:
        """Split markdown into {heading: body} by ## headers."""
        sections: Dict[str, str] = {}
        current_heading = "_preamble"
        current_lines: List[str] = []

        for line in content.splitlines():
            # Match ## heading (level 2 or 1)
            m = re.match(r"^#{1,3}\s+(.+)", line)
            if m:
                # Save previous section
                if current_lines:
                    sections[current_heading] = "\n".join(current_lines)
                current_heading = m.group(1).strip()
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_lines:
            sections[current_heading] = "\n".join(current_lines)

        return sections

    def _score_sentiment(self, text: str) -> float:
        """Score text sentiment from -1.0 (bearish) to +1.0 (bullish)."""
        text_lower = text.lower()
        bull_count = sum(1 for w in BULLISH_WORDS if w in text_lower)
        bear_count = sum(1 for w in BEARISH_WORDS if w in text_lower)
        total = bull_count + bear_count
        if total == 0:
            return 0.0
        return (bull_count - bear_count) / total

    def _parse_ticker_section(self, body: str, date_str: str,
                              signals: ResearchSignals) -> None:
        """Parse ticker mentions from a Tickers section."""
        # Match patterns like "- SMH: bullish, reason" or "SMH — strong"
        ticker_re = re.compile(
            r"[-*]?\s*([A-Z]{2,5})\s*[:—\-–]\s*(.+)",
        )
        for line in body.splitlines():
            m = ticker_re.match(line.strip())
            if m:
                symbol = m.group(1)
                context = m.group(2).strip()
                sentiment = self._score_sentiment(context)
                signals.ticker_mentions.append(
                    TickerMention(symbol=symbol, sentiment=sentiment,
                                 context=context, date=date_str)
                )

    def _parse_sector_section(self, body: str, date_str: str,
                              signals: ResearchSignals) -> None:
        """Parse sector signals from a Sectors section."""
        for line in body.splitlines():
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue
            sentiment = self._score_sentiment(line_stripped)
            # Find matching categories
            line_lower = line_stripped.lower()
            matched_cats: Set[str] = set()
            matched_keywords: List[str] = []
            for keyword, category in SECTOR_CATEGORY_MAP.items():
                if keyword in line_lower:
                    matched_cats.add(category)
                    matched_keywords.append(keyword)
            for cat in matched_cats:
                signals.sector_signals.append(
                    SectorSignal(category=cat, sentiment=sentiment,
                                 keywords=matched_keywords, date=date_str)
                )

    def _parse_macro_section(self, body: str, date_str: str,
                             signals: ResearchSignals) -> None:
        """Parse macro regime from a Macro section."""
        body_lower = body.lower()

        # Regime detection
        risk_on_words = {"risk-on", "risk on", "risk-on rally", "bullish", "rally",
                         "soft landing", "goldilocks", "expansion", "euphoria",
                         "upside", "recovery", "de-escalation", "ceasefire"}
        risk_off_words = {"risk-off", "risk off", "recession", "crisis",
                          "flight to safety", "defensive", "contraction",
                          "stagflation", "stagflationary", "pullback",
                          "correction", "drawdown", "sell-off"}
        on_score = sum(1 for w in risk_on_words if w in body_lower)
        off_score = sum(1 for w in risk_off_words if w in body_lower)
        if on_score > off_score:
            regime = "risk_on"
        elif off_score > on_score:
            regime = "risk_off"
        else:
            regime = "neutral"

        # Rate direction
        hawkish_words = {"hawkish", "rate hike", "tightening", "higher for longer",
                         "inflation", "elevated inflation", "inflation concerns"}
        dovish_words = {"dovish", "rate cut", "easing", "pivot", "lower rates",
                        "yields eased", "yields declined", "yields slipping"}
        hawk = sum(1 for w in hawkish_words if w in body_lower)
        dove = sum(1 for w in dovish_words if w in body_lower)
        if hawk > dove:
            rate_dir = "hawkish"
        elif dove > hawk:
            rate_dir = "dovish"
        else:
            rate_dir = "neutral"

        confidence = min(1.0, (on_score + off_score + hawk + dove) / 4)

        signals.macro_signals.append(
            MacroSignal(regime=regime, rate_direction=rate_dir,
                        confidence=confidence, date=date_str)
        )

    def _parse_themes_section(self, body: str,
                              signals: ResearchSignals) -> None:
        """Parse thematic keywords from a Themes section."""
        # Split on commas, newlines, or bullet points
        raw = re.split(r"[,\n]", body)
        for chunk in raw:
            theme = chunk.strip().lstrip("-*• ").strip()
            if theme and len(theme) > 2 and theme not in signals.themes:
                signals.themes.append(theme)

    def _extract_bold_tickers(self, body: str, date_str: str,
                              signals: ResearchSignals) -> None:
        """Extract tickers from **TICKER** markdown bold patterns.

        Real-world reports embed tickers as: **XLE** (description), **SMH**, etc.
        This is the primary extraction method for free-form research reports.
        """
        # Match **TICKER** optionally followed by parenthetical description
        bold_re = re.compile(r"\*\*([A-Z]{2,5})\*\*(?:\s*\(([^)]+)\))?")
        # Known ETF tickers (expand beyond just TICKER_CATEGORY_HINTS)
        from etf_screener import ALL_SEED_SYMBOLS
        known = set(TICKER_CATEGORY_HINTS.keys()) | set(ALL_SEED_SYMBOLS)
        # Also accept any bold uppercase that looks like a ticker
        # but filter out common false positives
        FALSE_POSITIVES = {
            "ETF", "ETFs", "AI", "EV", "US", "CEO", "CFO", "CTO",
            "IPO", "SEC", "CFTC", "GDP", "CPI", "PPI", "PMI",
            "YTD", "QoQ", "MoM", "YoY", "BPS", "ATH", "NFT",
            "DeFi", "LNG", "OPEC", "WTI", "BTC", "ETH", "SOL",
            "SVM", "GPU", "API", "LLM", "CHIPS", "TIPS",
        }

        already = {t.symbol for t in signals.ticker_mentions if t.date == date_str}

        for m in bold_re.finditer(body):
            symbol = m.group(1)
            if symbol in FALSE_POSITIVES or symbol in already:
                continue
            if symbol not in known and len(symbol) < 3:
                continue  # skip 2-letter unknowns

            # Get context: surrounding sentence or ±80 chars
            start = max(0, m.start() - 80)
            end = min(len(body), m.end() + 80)
            context = body[start:end].replace("\n", " ").strip()
            # Include parenthetical if present
            paren = m.group(2)
            if paren:
                context = f"{paren} | {context}"

            sentiment = self._score_sentiment(context)
            signals.ticker_mentions.append(
                TickerMention(symbol=symbol, sentiment=sentiment,
                              context=context[:120], date=date_str)
            )
            already.add(symbol)

    def _deduplicate_mentions(self, signals: ResearchSignals,
                              date_str: str) -> None:
        """Deduplicate ticker mentions for a given date.

        Keep the mention with highest absolute sentiment per symbol per date.
        """
        best: Dict[str, TickerMention] = {}
        other_dates: List[TickerMention] = []

        for mention in signals.ticker_mentions:
            if mention.date != date_str:
                other_dates.append(mention)
                continue
            key = mention.symbol
            if key not in best or abs(mention.sentiment) > abs(best[key].sentiment):
                best[key] = mention

        signals.ticker_mentions = other_dates + list(best.values())

    def _scan_for_tickers(self, body: str, date_str: str,
                          signals: ResearchSignals) -> None:
        """Scan arbitrary text for uppercase ticker-like mentions."""
        # Only pick up tickers we know about (avoid false positives)
        known = set(TICKER_CATEGORY_HINTS.keys())
        words = re.findall(r"\b([A-Z]{2,5})\b", body)
        already_mentioned = {t.symbol for t in signals.ticker_mentions
                             if t.date == date_str}
        for word in words:
            if word in known and word not in already_mentioned:
                # Get surrounding context (±30 chars)
                idx = body.find(word)
                context = body[max(0, idx - 30):idx + len(word) + 30].strip()
                sentiment = self._score_sentiment(context)
                signals.ticker_mentions.append(
                    TickerMention(symbol=word, sentiment=sentiment,
                                 context=context, date=date_str)
                )
                already_mentioned.add(word)

    # ------------------------------------------------------------------
    # Bias weight computation (for ETF screening integration)
    # ------------------------------------------------------------------

    def compute_bias_weights(
        self, signals: ResearchSignals, decay: float = 0.85,
    ) -> Dict[str, float]:
        """Convert research signals into per-symbol bias weights.

        Returns {symbol: weight} where weight is in [-1.0, +1.0].
        More recent mentions get higher weight (exponential decay).

        These weights are added to the ETF screener's composite score
        to bias the ranking toward research-informed ideas.

        Args:
            signals: Aggregated research signals
            decay: Per-day decay factor (0.85 = 15% decay per day older)
        """
        if not signals.dates_read:
            return {}

        # Date ordering for decay: most recent = index 0
        date_order = sorted(signals.dates_read, reverse=True)
        date_rank = {d: i for i, d in enumerate(date_order)}

        # Ticker-level bias
        symbol_scores: Dict[str, List[float]] = {}
        for mention in signals.ticker_mentions:
            rank = date_rank.get(mention.date, len(date_order))
            weight = mention.sentiment * (decay ** rank)
            symbol_scores.setdefault(mention.symbol, []).append(weight)

        # Sector-level bias → spread to known tickers in that category
        cat_scores: Dict[str, List[float]] = {}
        for sig in signals.sector_signals:
            rank = date_rank.get(sig.date, len(date_order))
            weight = sig.sentiment * (decay ** rank) * 0.5  # half weight for sector-level
            cat_scores.setdefault(sig.category, []).append(weight)

        # Apply sector scores to tickers
        for symbol, category in TICKER_CATEGORY_HINTS.items():
            if category in cat_scores:
                symbol_scores.setdefault(symbol, []).extend(cat_scores[category])

        # Macro regime → broad market bias
        for macro in signals.macro_signals:
            rank = date_rank.get(macro.date, len(date_order))
            if macro.regime == "risk_on":
                macro_bias = 0.3 * macro.confidence * (decay ** rank)
            elif macro.regime == "risk_off":
                macro_bias = -0.3 * macro.confidence * (decay ** rank)
            else:
                macro_bias = 0.0
            # Apply to broad market tickers
            for sym in ("SPY", "QQQ", "IWM"):
                symbol_scores.setdefault(sym, []).append(macro_bias)
            # Dovish → positive for gold, TLT
            if macro.rate_direction == "dovish":
                rate_bias = 0.2 * macro.confidence * (decay ** rank)
                for sym in ("GLD", "SLV", "TLT"):
                    symbol_scores.setdefault(sym, []).append(rate_bias)
            elif macro.rate_direction == "hawkish":
                rate_bias = -0.15 * macro.confidence * (decay ** rank)
                for sym in ("GLD", "SLV", "TLT"):
                    symbol_scores.setdefault(sym, []).append(rate_bias)

        # Average scores per symbol, clamp to [-1, 1]
        result: Dict[str, float] = {}
        for symbol, scores in symbol_scores.items():
            avg = sum(scores) / len(scores) if scores else 0.0
            result[symbol] = max(-1.0, min(1.0, avg))

        return result

    def compute_category_bias(
        self, signals: ResearchSignals,
    ) -> Dict[str, float]:
        """Compute category-level bias weights.

        Returns {category: weight} for use in screening when specific
        tickers aren't mentioned but sector signals exist.
        """
        cat_scores: Dict[str, List[float]] = {}
        for sig in signals.sector_signals:
            cat_scores.setdefault(sig.category, []).append(sig.sentiment)

        return {
            cat: sum(scores) / len(scores)
            for cat, scores in cat_scores.items()
            if scores
        }


def main() -> None:
    """CLI: print extracted signals from recent research."""
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Research reader — extract signals from daily reports")
    parser.add_argument("--days", type=int, default=7, help="Number of days to read (default: 7)")
    parser.add_argument("--dir", type=str, default=RESEARCH_DIR, help="Research directory")
    args = parser.parse_args()

    reader = ResearchReader(research_dir=args.dir)
    signals = reader.read_last_n_days(args.days)

    if not signals.dates_read:
        print("\n  No research files found.\n")
        return

    print(f"\n{'='*65}")
    print(f"  RESEARCH SIGNALS — {len(signals.dates_read)} reports")
    print(f"  Dates: {', '.join(signals.dates_read)}")
    print(f"{'='*65}")

    if signals.macro_signals:
        print("\n  Macro Regime:")
        for m in signals.macro_signals:
            print(f"    [{m.date}] regime={m.regime}, rates={m.rate_direction}, "
                  f"confidence={m.confidence:.0%}")

    if signals.sector_signals:
        print("\n  Sector Signals:")
        for s in signals.sector_signals:
            direction = "BULL" if s.sentiment > 0 else "BEAR" if s.sentiment < 0 else "NEUT"
            print(f"    [{s.date}] {s.category:<16} {direction} ({s.sentiment:+.2f}) "
                  f"keys={s.keywords}")

    if signals.ticker_mentions:
        print("\n  Ticker Mentions:")
        for t in signals.ticker_mentions:
            direction = "BULL" if t.sentiment > 0 else "BEAR" if t.sentiment < 0 else "NEUT"
            print(f"    [{t.date}] {t.symbol:<6} {direction} ({t.sentiment:+.2f}) "
                  f"{t.context[:60]}")

    if signals.themes:
        print(f"\n  Themes: {', '.join(signals.themes)}")

    # Compute and show bias weights
    bias = reader.compute_bias_weights(signals)
    if bias:
        print(f"\n  Bias Weights (top 15):")
        sorted_bias = sorted(bias.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        for sym, w in sorted_bias:
            bar = "+" * int(abs(w) * 20) if w > 0 else "-" * int(abs(w) * 20)
            print(f"    {sym:<6} {w:+.3f}  {bar}")

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
