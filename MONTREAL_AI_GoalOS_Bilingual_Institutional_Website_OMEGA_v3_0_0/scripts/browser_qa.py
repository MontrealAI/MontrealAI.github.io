#!/usr/bin/env python3
"""Render every public page without a network server.

Some controlled QA environments block Chromium HTTP/file navigation. This runner creates a
self-contained render document for each page by inlining local CSS, JavaScript and image assets,
then uses page.set_content(). It therefore tests the actual release payload while generating no
external network traffic.
"""
from __future__ import annotations
import asyncio, base64, datetime, json, mimetypes, pathlib, re, sys
from urllib.parse import urlsplit
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "qa"
OUT = OUT_DIR / "BROWSER_QA_REPORT.json"
EXCLUDE = {"START_HERE.html"}
EXCLUDE_PREFIXES = ("goalos-documents/", "BUILD_SOURCE/", "qa/")
PAGES = sorted(
    p.relative_to(ROOT).as_posix()
    for p in ROOT.rglob("*.html")
    if p.relative_to(ROOT).as_posix() not in EXCLUDE
    and "_Downloads" not in p.as_posix()
    and not p.relative_to(ROOT).as_posix().startswith(EXCLUDE_PREFIXES)
)
VIEWPORTS = {
    "desktop": {"width": 1440, "height": 1000},
    "mobile": {"width": 390, "height": 844},
}
SHOTS = {
    ("index.html", "desktop"): "home_desktop.png",
    ("index.html", "mobile"): "home_mobile.png",
    ("en/index.html", "desktop"): "home_english_desktop.png",
    ("goalos/index.html", "desktop"): "goalos_desktop.png",
    ("proof-missions/index.html", "desktop"): "proof_missions_desktop.png",
    ("proof-missions/ai-deployment/index.html", "desktop"): "deployment_mission_desktop.png",
    ("en/proof-missions/rfp-to-revenue/index.html", "mobile"): "rfp_mission_english_mobile.png",
    ("proof/index.html", "desktop"): "proof_desktop.png",
    ("company/index.html", "desktop"): "company_desktop.png",
    ("goalos-legal/index.html", "mobile"): "legal_mobile.png",
    ("contact/index.html", "mobile"): "contact_mobile.png",
    ("business-model/index.html", "desktop"): "business_model_desktop.png",
    ("mission-lab/index.html", "mobile"): "mission_lab_mobile.png",
    ("alpha-agi-ascension/index.html", "desktop"): "alpha_ascension_desktop.png",
}


def local_path(page_file: pathlib.Path, ref: str) -> pathlib.Path | None:
    if not ref or ref.startswith(("data:", "http://", "https://", "mailto:", "tel:", "#", "javascript:")):
        return None
    split = urlsplit(ref)
    raw = split.path
    if not raw:
        return None
    target = ROOT / raw.lstrip("/") if raw.startswith("/") else page_file.parent / raw
    target = target.resolve()
    try:
        target.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return target


def data_uri(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def inline_css(css_text: str, css_file: pathlib.Path) -> str:
    pattern = re.compile(r"url\((['\"]?)([^)'\"]+)\1\)")
    def repl(m: re.Match[str]) -> str:
        ref = m.group(2).strip()
        target = local_path(css_file, ref)
        if target and target.is_file():
            return f"url('{data_uri(target)}')"
        return m.group(0)
    return pattern.sub(repl, css_text)


def make_render_document(rel: str) -> tuple[str, list[str]]:
    page_file = ROOT / rel
    soup = BeautifulSoup(page_file.read_text(encoding="utf-8"), "html.parser")
    preparation_errors: list[str] = []

    # The release CSP is verified statically. Remove it only from the synthetic QA document so
    # inlined assets can render in the restricted browser runner.
    for meta in soup.find_all("meta"):
        if str(meta.get("http-equiv", "")).lower() == "content-security-policy":
            meta.decompose()

    for link in list(soup.find_all("link", rel=lambda v: v and "stylesheet" in v)):
        href = link.get("href", "")
        target = local_path(page_file, href)
        if target and target.is_file():
            style = soup.new_tag("style")
            style.string = inline_css(target.read_text(encoding="utf-8"), target)
            link.replace_with(style)
        else:
            preparation_errors.append(f"stylesheet not resolved: {href}")

    for script in list(soup.find_all("script", src=True)):
        src = script.get("src", "")
        target = local_path(page_file, src)
        if target and target.is_file():
            repl = soup.new_tag("script")
            repl.string = target.read_text(encoding="utf-8")
            script.replace_with(repl)
        else:
            preparation_errors.append(f"script not resolved: {src}")

    for img in soup.find_all("img", src=True):
        src = img.get("src", "")
        target = local_path(page_file, src)
        if target and target.is_file():
            img["src"] = data_uri(target)
        elif not src.startswith("data:"):
            preparation_errors.append(f"image not resolved: {src}")

    # Make absolute same-origin links inert for QA; navigation is not part of page rendering.
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if href.startswith("/"):
            a["data-original-href"] = href
            a["href"] = "#"

    return str(soup), preparation_errors


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-background-networking"],
        )
        for vp_name, vp in VIEWPORTS.items():
            context = await browser.new_context(viewport=vp, device_scale_factor=1, locale="fr-CA")
            page = await context.new_page()
            state = {"errors": []}
            page.on("pageerror", lambda exc: state["errors"].append("pageerror: " + str(exc)))
            page.on("console", lambda msg: state["errors"].append(f"console {msg.type}: {msg.text}") if msg.type == "error" else None)
            for i, page_path in enumerate(PAGES, 1):
                state["errors"] = []
                html, prep_errors = make_render_document(page_path)
                state["errors"].extend(prep_errors)
                try:
                    await page.set_content(html, wait_until="domcontentloaded", timeout=10000)
                    await page.wait_for_timeout(5)
                    data = await page.evaluate("""() => ({
                      title: document.title,
                      lang: document.documentElement.lang,
                      h1Count: document.querySelectorAll('h1').length,
                      overflow: Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth),
                      brokenImages: [...document.images].filter(i => i.complete && i.naturalWidth === 0).map(i => i.getAttribute('alt') || i.getAttribute('src')),
                      menuButton: !!document.querySelector('.menu-btn'),
                      menuNav: !!document.querySelector('.nav'),
                      main: !!document.querySelector('main'),
                      footer: !!document.querySelector('footer'),
                      textLength: (document.body.innerText || '').trim().length
                    })""")
                    menu_ok = True
                    if vp_name == "mobile" and data["menuButton"] and data["menuNav"]:
                        try:
                            menu_ok = await page.evaluate("""() => { const b=document.querySelector('.menu-btn'), n=document.querySelector('.nav'); if(!b||!n)return false; b.click(); const ok=n.classList.contains('open') && b.getAttribute('aria-expanded')==='true'; b.click(); return ok && !n.classList.contains('open'); }""")
                        except Exception as exc:
                            menu_ok = False
                            state["errors"].append("menu: " + str(exc))
                    shot = SHOTS.get((page_path, vp_name))
                    if shot:
                        await page.screenshot(path=str(OUT_DIR / shot), full_page=True, timeout=30000)
                    result = {
                        "page": page_path,
                        "viewport": vp_name,
                        "rendered": True,
                        "title": data["title"],
                        "lang": data["lang"],
                        "h1_count": data["h1Count"],
                        "overflow_px": data["overflow"],
                        "broken_images": data["brokenImages"],
                        "menu_ok": menu_ok,
                        "has_main": data["main"],
                        "has_footer": data["footer"],
                        "text_length": data["textLength"],
                        "errors": list(state["errors"]),
                    }
                except Exception as exc:
                    result = {
                        "page": page_path,
                        "viewport": vp_name,
                        "rendered": False,
                        "title": "",
                        "lang": "",
                        "h1_count": 0,
                        "overflow_px": 0,
                        "broken_images": [],
                        "menu_ok": False,
                        "has_main": False,
                        "has_footer": False,
                        "text_length": 0,
                        "errors": list(state["errors"]) + [str(exc)],
                    }
                results.append(result)
                if i % 10 == 0 or i == len(PAGES):
                    print(f"{vp_name}: {i}/{len(PAGES)}", flush=True)
            await page.close()
            await context.close()
        await browser.close()

    report = {
        "schema": "montrealai.goalos.browser-qa.v3",
        "method": "network-free set_content with local CSS, JS and image assets inlined",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pages_tested": len(PAGES),
        "scenarios": len(results),
        "render_failures": [r for r in results if not r["rendered"]],
        "runtime_error_scenarios": [r for r in results if r["errors"]],
        "overflow_failures": [r for r in results if r["overflow_px"] > 0],
        "broken_image_scenarios": [r for r in results if r["broken_images"]],
        "language_failures": [r for r in results if not r["lang"]],
        "h1_failures": [r for r in results if r["h1_count"] != 1],
        "structure_failures": [r for r in results if not r["has_main"] or not r["has_footer"] or r["text_length"] < 100],
        "mobile_menu_failures": [r for r in results if r["viewport"] == "mobile" and not r["menu_ok"]],
        "results": results,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {k: (len(v) if isinstance(v, list) else v) for k, v in report.items() if k != "results"}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    failed = any(summary[k] for k in (
        "render_failures", "runtime_error_scenarios", "overflow_failures", "broken_image_scenarios",
        "language_failures", "h1_failures", "structure_failures", "mobile_menu_failures"
    ))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
