#!/usr/bin/env python3
"""
Single-application RSS aggregator with AI rewrite/translation.

One repo = one application. This script processes exactly ONE feed list
(named by `app_name` in config.yml) and produces ONE output feed.

Each run:
  1. Loads settings from config.yml.
  2. Reads the feed list feeds/<app_name>.json (staged from the private repo).
  3. Fetches every source feed (parallel, retries) and merges new entries
     into data/<app_name>.json, de-duplicating by GUID/link.
  4. Prunes the archive to retention_days.
  5. Runs an AI rewrite/translation pass on NEW items (up to
     rewrites_per_run; overflow carries to the next run; results cached).
  6. Writes public/feed.xml (+ index.html).

output_html controls the description:
  false -> clean text  (AI-rewritten if enabled, else stripped original)
  true  -> original source HTML is preserved untouched; only the TITLE is
           AI-translated/rewritten (saves cost, keeps rich formatting).

The OpenAI key is read from OPENAI_API_KEY (a GitHub Secret), never a file.
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import hashlib
import html
import json
import os
import re
import sys
from email.utils import format_datetime
from html.parser import HTMLParser
from pathlib import Path
from xml.sax.saxutils import escape

import feedparser
import requests
import yaml

ROOT = Path(__file__).resolve().parent
FEEDS_DIR = ROOT / "feeds"
DATA_DIR = ROOT / "data"
PUBLIC_DIR = ROOT / "public"
CONFIG_PATH = ROOT / "config.yml"

NOW = dt.datetime.now(dt.timezone.utc)
USER_AGENT = "Mozilla/5.0 (compatible; RSS-Aggregator/3.0; +https://github.com/)"


def load_config() -> dict:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    d = {
        "app_name": "app", "feed_title": "Feed",
        "output_html": False, "max_output_items": 10000, "retention_days": 365,
        "site_title": "My RSS Feed", "site_base_url": "",
        "fetch_workers": 12, "fetch_timeout": 30, "fetch_retries": 2,
        "translate_enabled": False, "openai_model": "chatgpt-4o-latest",
        "openai_base_url": "https://api.openai.com/v1",
        "target_language": "English", "rewrites_per_run": 1000,
        "translate_batch_size": 10, "translate_workers": 6,
        "translate_timeout": 90, "translate_max_retries": 2,
        "rewrite_prompt": "Translate to English and rewrite clearly.",
    }
    d.update({k: v for k, v in cfg.items() if v is not None})
    if os.environ.get("SITE_BASE_URL"):
        d["site_base_url"] = os.environ["SITE_BASE_URL"]
    if os.environ.get("APP_NAME"):
        d["app_name"] = os.environ["APP_NAME"]
    d["site_base_url"] = str(d["site_base_url"]).rstrip("/")
    return d


CFG = load_config()
CUTOFF = NOW - dt.timedelta(days=int(CFG["retention_days"]))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
APP = CFG["app_name"]


def log(m: str) -> None:
    print(m, flush=True)


# ---- helpers ----
def to_utc_iso(st):
    if not st:
        return None
    try:
        return dt.datetime(*st[:6], tzinfo=dt.timezone.utc).isoformat()
    except (ValueError, TypeError):
        return None


def parse_iso(v):
    if not v:
        return None
    try:
        d = dt.datetime.fromisoformat(v)
        return d if d.tzinfo else d.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def stable_id(entry, url):
    for k in ("id", "guid"):
        if entry.get(k):
            return f"id:{entry[k]}"
    if entry.get("link"):
        return f"link:{entry['link']}"
    basis = url + "|" + entry.get("title", "") + "|" + (
        entry.get("published", "") or entry.get("updated", ""))
    return "hash:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()


class _Stripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, d):
        self.parts.append(d)


def strip_html(s):
    if not s:
        return ""
    p = _Stripper()
    try:
        p.feed(s)
    except Exception:
        return re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", " ".join(p.parts)).strip()


# ---- fetching ----
def fetch_one(source):
    url = source["url"]
    last = None
    for _ in range(int(CFG["fetch_retries"]) + 1):
        try:
            r = requests.get(url, timeout=int(CFG["fetch_timeout"]),
                             headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
            items = []
            for e in parsed.entries:
                pub = to_utc_iso(e.get("published_parsed")) or \
                    to_utc_iso(e.get("updated_parsed"))
                it = {
                    "id": stable_id(e, url),
                    "title": (e.get("title") or "").strip() or "(no title)",
                    "link": (e.get("link") or "").strip(),
                    "summary": (e.get("summary") or "").strip(),
                    "published": pub,
                    "source": source.get("name", "source"),
                }
                for m in ("country", "platform", "language"):
                    if source.get(m):
                        it[m] = source[m]
                items.append(it)
            log(f"  ok   {len(items):4d}  {source.get('name','')[:46]}")
            return source, items
        except Exception as exc:  # noqa: BLE001
            last = exc
    log(f"  FAIL  ---  {source.get('name','')[:46]}  ({last})")
    return source, []


# ---- archive ----
def load_archive():
    p = DATA_DIR / f"{APP}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"  warning: {p} corrupt, starting fresh")
    return {"app": APP, "items": {}}


def save_archive(archive):
    for it in archive.get("items", {}).values():
        it.pop("source_url", None)   # never expose feed URLs
    (DATA_DIR / f"{APP}.json").write_text(
        json.dumps(archive, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")


def ref_date(it):
    return parse_iso(it.get("published")) or parse_iso(it.get("first_seen")) or NOW


# ---- AI rewrite ----
def _openai_chat(messages):
    url = f"{CFG['openai_base_url'].rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": CFG["openai_model"], "messages": messages,
               "temperature": 0.3}
    last = None
    for _ in range(int(CFG["translate_max_retries"]) + 1):
        try:
            r = requests.post(url, headers=headers, json=payload,
                              timeout=int(CFG["translate_timeout"]))
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"OpenAI call failed: {last}")


def _parse_json_block(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    a, b = text.find("["), text.rfind("]")
    if a != -1 and b != -1:
        text = text[a:b + 1]
    return json.loads(text)


def translate_batch(batch, want_desc):
    payload = []
    for it in batch:
        row = {"id": it["id"], "title": it.get("title", "")}
        if want_desc:
            row["text"] = strip_html(it.get("summary", ""))[:4000]
        payload.append(row)
    fields = ('"title": <rewritten title>'
              + (', "text": <rewritten description>' if want_desc else ""))
    system = (
        CFG["rewrite_prompt"].strip()
        + f"\n\nTarget language: {CFG['target_language']}."
        + "\n\nYou will receive a JSON array of items. Return ONLY a JSON "
        "array of objects with the same ids, each as "
        f"{{\"id\": <id>, {fields}}}. JSON only, no commentary."
    )
    content = _openai_chat([{"role": "system", "content": system},
                            {"role": "user",
                             "content": json.dumps(payload, ensure_ascii=False)}])
    out = {}
    for o in _parse_json_block(content):
        if isinstance(o, dict) and o.get("id"):
            out[str(o["id"])] = {"title": (o.get("title") or "").strip(),
                                 "summary": (o.get("text") or "").strip()}
    return out


def run_translation(archive):
    if not CFG["translate_enabled"]:
        log("=== translation disabled ==="); return
    if not OPENAI_API_KEY:
        log("=== translation skipped: OPENAI_API_KEY not set ==="); return
    want_desc = not bool(CFG["output_html"])
    pending = [it for it in archive["items"].values() if not it.get("ai_done")]
    pending.sort(key=ref_date, reverse=True)
    todo = pending[:int(CFG["rewrites_per_run"])]
    if not todo:
        log("=== translation: nothing new to rewrite ==="); return
    log(f"=== translation: {len(todo)} of {len(pending)} pending, "
        f"model={CFG['openai_model']}, desc={'yes' if want_desc else 'title-only'} ===")
    bs = max(1, int(CFG["translate_batch_size"]))
    batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]

    def work(b):
        try:
            res = translate_batch(b, want_desc)
        except Exception as exc:  # noqa: BLE001
            log(f"  batch failed ({exc}); will retry next run"); return 0
        n = 0
        for it in b:
            r = res.get(it["id"])
            if r and (r["title"] or r["summary"]):
                it["ai_title"] = r["title"] or it.get("title")
                if want_desc:
                    it["ai_summary"] = r["summary"] or strip_html(it.get("summary"))
                it["ai_done"] = True
                n += 1
        return n

    done = 0
    with cf.ThreadPoolExecutor(max_workers=int(CFG["translate_workers"])) as pool:
        for n in pool.map(work, batches):
            done += n
    log(f"  rewritten this run: {done}")


# ---- output ----
def rss_date(d):
    return format_datetime(d)


_ILLEGAL = re.compile("[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def xclean(v):
    return "" if v is None else _ILLEGAL.sub("", str(v))


def xe(v):
    return escape(xclean(v))


def out_title(it):
    return it.get("ai_title") or it.get("title", "")


def out_desc(it):
    if CFG["output_html"]:
        return it.get("summary", "")          # original HTML preserved
    return it.get("ai_summary") or strip_html(it.get("summary", ""))


def build_rss(items):
    base = CFG["site_base_url"]
    self_url = f"{base}/feed.xml" if base else ""
    p = ['<?xml version="1.0" encoding="UTF-8"?>',
         '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">',
         "<channel>",
         f"<title>{xe(CFG['site_title'])}</title>",
         f"<description>{xe(CFG['feed_title'])}</description>",
         "<language>en</language>",
         f"<lastBuildDate>{rss_date(NOW)}</lastBuildDate>",
         "<generator>rss-aggregator</generator>"]
    if self_url:
        p.append(f"<link>{xe(self_url)}</link>")
        p.append(f'<atom:link href="{xe(self_url)}" rel="self" '
                 'type="application/rss+xml"/>')
    else:
        p.append("<link>https://example.com</link>")
    for it in items:
        p.append("<item>")
        p.append(f"<title>{xe(out_title(it))}</title>")
        if it.get("link"):
            p.append(f"<link>{xe(it['link'])}</link>")
        gid = it.get("link") or it["id"]
        p.append(f'<guid isPermaLink="{"true" if it.get("link") else "false"}">'
                 f'{xe(gid)}</guid>')
        p.append(f"<pubDate>{rss_date(ref_date(it))}</pubDate>")
        p.append(f"<category>{xe(it['source'])}</category>")
        for m in ("country", "platform", "language"):
            if it.get(m):
                p.append(f"<category>{xe(it[m])}</category>")
        desc = out_desc(it)
        if desc:
            p.append(f"<description><![CDATA[{xclean(desc).replace(']]>', ']]&gt;')}]]></description>")
        p.append("</item>")
    p.append("</channel>")
    p.append("</rss>")
    return "\n".join(p)


def build_index(count):
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(CFG['site_title'])}</title>
<style>body{{font-family:system-ui,Arial,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}}
a{{color:#0b66c3}}</style></head><body>
<h1>{html.escape(CFG['site_title'])}</h1>
<p>Feed: <a href="feed.xml">feed.xml</a></p>
<p><small>Updated {html.escape(NOW.strftime('%Y-%m-%d %H:%M UTC'))} ·
{count} items published · retention {CFG['retention_days']}d</small></p>
</body></html>"""
    (PUBLIC_DIR / "index.html").write_text(page, encoding="utf-8")
    (PUBLIC_DIR / ".nojekyll").write_text("", encoding="utf-8")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    PUBLIC_DIR.mkdir(exist_ok=True)
    feed_file = FEEDS_DIR / f"{APP}.json"
    if not feed_file.exists():
        log(f"ERROR: feed list {feed_file} not found "
            f"(is app_name '{APP}' correct and staged from the private repo?)")
        return 1
    conf = json.loads(feed_file.read_text(encoding="utf-8"))
    sources = conf.get("sources", [])
    log(f"=== app '{APP}' ({len(sources)} sources) ===")

    archive = load_archive()
    items = archive.get("items", {})
    new = 0
    with cf.ThreadPoolExecutor(max_workers=int(CFG["fetch_workers"])) as pool:
        for _s, got in pool.map(fetch_one, sources):
            for it in got:
                ex = items.get(it["id"])
                if ex:
                    it["first_seen"] = ex.get("first_seen")
                    for k in ("ai_title", "ai_summary", "ai_done"):
                        if ex.get(k) is not None:
                            it[k] = ex[k]
                    items[it["id"]] = it
                else:
                    it["first_seen"] = NOW.isoformat()
                    items[it["id"]] = it
                    new += 1
    before = len(items)
    items = {k: v for k, v in items.items() if ref_date(v) >= CUTOFF}
    pruned = before - len(items)
    archive.update({"app": APP, "items": items, "updated": NOW.isoformat()})
    log(f"  new: {new}  pruned: {pruned}  archive: {len(items)}")

    run_translation(archive)
    save_archive(archive)

    ordered = sorted(items.values(), key=ref_date, reverse=True)
    published = ordered[:int(CFG["max_output_items"])]
    (PUBLIC_DIR / "feed.xml").write_text(build_rss(published), encoding="utf-8")
    build_index(len(published))
    log(f"=== done: published {len(published)} items to public/feed.xml ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
