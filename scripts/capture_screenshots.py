"""
Capture screenshots for both Streamlit and HTML dashboards.
Saves all screenshots to: dashboard_screenshots/
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "dashboard_screenshots"
HTML_OUT = OUTPUT_DIR / "html_dashboard"
STREAMLIT_OUT = OUTPUT_DIR / "streamlit_dashboard"

HTML_OUT.mkdir(parents=True, exist_ok=True)
STREAMLIT_OUT.mkdir(parents=True, exist_ok=True)

def capture_html_dashboard(browser):
    print("Capturing HTML Dashboard screenshots...")
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    html_path = (ROOT / "dashboard" / "index.html").as_uri()
    page.goto(html_path, wait_until="networkidle")
    time.sleep(1)

    tabs = [
        ("overview", "01_overview.png"),
        ("sources", "02_source_performance.png"),
        ("funnel", "03_pipeline_funnel.png"),
        ("ml", "04_ml_model.png"),
        ("explainability", "05_explainability.png"),
        ("ai", "06_ai_insights.png"),
    ]

    for tab_id, filename in tabs:
        print(f"  Switching to HTML tab: {tab_id}...")
        page.evaluate(f"showPage('{tab_id}')")
        time.sleep(1)
        out_file = HTML_OUT / filename
        page.screenshot(path=str(out_file), full_page=True)
        print(f"  Saved: {out_file.name} ({out_file.stat().st_size // 1024} KB)")

    page.close()


def capture_streamlit_dashboard(browser):
    print("\nCapturing Streamlit Dashboard screenshots...")
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    url = "http://localhost:8501"
    page.goto(url, wait_until="networkidle")
    
    # Wait for Streamlit to finish initial load
    page.wait_for_selector("[data-testid='stSidebar']", timeout=15000)
    time.sleep(3)

    views = [
        ("📋 Overview", "01_overview.png"),
        ("📈 Performance", "02_performance.png"),
        ("🔬 Explainability", "03_explainability.png"),
        ("🤖 AI Insights", "04_ai_insights.png"),
        ("🎯 What-If Scorer", "05_what_if_scorer.png"),
    ]

    for label, filename in views:
        safe_label = label.encode('ascii', 'ignore').decode('ascii').strip()
        print(f"  Selecting Streamlit view: {safe_label}...")
        try:
            # Click the radio button label in sidebar
            radio_option = page.locator(f"label:has-text('{label}')")
            radio_option.first.click()
            time.sleep(2.5) # Allow Plotly / Streamlit to update and render
            out_file = STREAMLIT_OUT / filename
            page.screenshot(path=str(out_file), full_page=True)
            print(f"  Saved: {out_file.name} ({out_file.stat().st_size // 1024} KB)")
        except Exception as e:
            print(f"  Error on {safe_label}: {e}")

    page.close()


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        try:
            capture_html_dashboard(browser)
            capture_streamlit_dashboard(browser)
        finally:
            browser.close()
    print("\nAll screenshots captured successfully!")

if __name__ == "__main__":
    main()
