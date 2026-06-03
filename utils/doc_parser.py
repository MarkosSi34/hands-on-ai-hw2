import re
import sys
import logging
import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DocumentParser:
    """
    Fetch a URL, extract its readable text, and persist it as .txt or .pdf.

    One instance == one document. Call run() to execute the full
    fetch → extract → save pipeline and get back the written file path.
    """

    # Polite, non-bot-looking UA — some sites (incl. Wikipedia) block the
    # default python-requests UA.
    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "doc_parser/1.0 (hands-on-ai-hw2 knowledge-base builder)"
    )
    REQUEST_TIMEOUT = 30  # seconds

    # Tags that never carry article content.
    _STRIP_TAGS = [
        "script", "style", "noscript", "nav", "header", "footer", "aside",
        "form", "figure", "figcaption", "table", "sup",
    ]
    # Tags whose text we keep, in document order.
    _CONTENT_TAGS = ["p", "h2", "h3", "h4", "li"]

    # Unicode font for PDF output (present on most Linux installs).
    _FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    _FONT_NAME = "DejaVuSans"

    def __init__(self, url: str, dest_dir: str):
        self.url = url
        self.dest_dir = Path(dest_dir)
        self.title: str = ""
        self.text: str = ""

    # Fetch
    def fetch(self):
        """Download the raw HTML for self.url."""
        logging.info(f"Fetching: {self.url}")
        resp = requests.get(
            self.url,
            headers={"User-Agent": self.USER_AGENT},
            timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding or resp.encoding
        logging.info(f"Fetched {len(resp.text):,} bytes (HTTP {resp.status_code}).")
        return resp.text

    # Extract
    def extract(self, html: str):
        """
        Parse HTML and populate self.title and self.text with clean,
        paragraph-separated reading text.
        """
        soup = BeautifulSoup(html, "lxml")

        self.title = self._extract_title(soup)

        # Pick the most content-rich region of the page.
        body = (
            soup.select_one(".mw-parser-output")   # Wikipedia / MediaWiki
            or soup.find("article")
            or soup.find("main")
            or soup.body
            or soup
        )

        # Drop boilerplate tags from the chosen region.
        for tag in body.find_all(self._STRIP_TAGS):
            tag.decompose()

        # Collect text from content tags in document order.
        blocks = []
        for el in body.find_all(self._CONTENT_TAGS):
            chunk = el.get_text(" ", strip=True)
            chunk = self._clean(chunk)
            if len(chunk) >= 30:  # skip nav crumbs, empty list items, etc.
                blocks.append(chunk)

        if not blocks:
            # Last-ditch fallback: whole-page text.
            blocks = [self._clean(body.get_text(" ", strip=True))]

        self.text = "\n\n".join(blocks).strip()
        logging.info(
            f"Extracted title='{self.title}' | "
            f"{len(blocks)} block(s), {len(self.text):,} chars."
        )
        if not self.text:
            raise RuntimeError(f"No readable text extracted from {self.url}")

    def _extract_title(self, soup: BeautifulSoup):
        """Prefer the MediaWiki/H1 heading, fall back to <title>."""
        h1 = soup.select_one("#firstHeading") or soup.find("h1")
        if h1 and h1.get_text(strip=True):
            return h1.get_text(strip=True)
        if soup.title and soup.title.get_text(strip=True):
            # Trim trailing " - Wikipedia" / " | Site" suffixes.
            return re.split(r"\s+[-|–]\s+", soup.title.get_text(strip=True))[0]
        return "document"

    @staticmethod
    def _clean(text: str):
        """Strip reference markers, edit links, and collapse whitespace."""
        text = re.sub(r"\[\s*edit\s*\]", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\d+\]", "", text)          # [1], [23]
        text = re.sub(r"\[[a-z]\]", "", text)         # [a], [b] footnotes
        text = text.replace("\xa0", " ")              # non-breaking spaces
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    # Filenames
    @staticmethod
    def _slugify(title: str):
        slug = title.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_-]+", "_", slug)
        return slug.strip("_") or "document"

    # Save
    def save_txt(self):
        """Write the extracted text as a UTF-8 .txt file."""
        out = self.dest_dir / f"{self._slugify(self.title)}.txt"
        header = f"# {self.title}\n# Source: {self.url}\n\n"
        out.write_text(header + self.text + "\n", encoding="utf-8")
        return out

    def save_pdf(self):
        """Render the extracted text as a simple multi-page .pdf file."""
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        # Register a Unicode font if available, else fall back to Helvetica.
        font = "Helvetica"
        if Path(self._FONT_PATH).exists():
            try:
                pdfmetrics.registerFont(TTFont(self._FONT_NAME, self._FONT_PATH))
                font = self._FONT_NAME
            except Exception as e:  # pragma: no cover - font edge cases
                logging.warning(f"Could not register {self._FONT_NAME}: {e}")

        styles = getSampleStyleSheet()
        body_style = styles["BodyText"]
        title_style = styles["Title"]
        body_style.fontName = font
        title_style.fontName = font
        body_style.leading = 14

        out = self.dest_dir / f"{self._slugify(self.title)}.pdf"
        doc = SimpleDocTemplate(
            str(out), pagesize=A4,
            leftMargin=2 * cm, rightMargin=2 * cm,
            topMargin=2 * cm, bottomMargin=2 * cm,
            title=self.title,
        )

        flow = [Paragraph(self._escape(self.title), title_style), Spacer(1, 6)]
        flow.append(Paragraph(self._escape(f"Source: {self.url}"), body_style))
        flow.append(Spacer(1, 12))
        for para in self.text.split("\n\n"):
            flow.append(Paragraph(self._escape(para), body_style))
            flow.append(Spacer(1, 6))

        doc.build(flow)
        return out

    @staticmethod
    def _escape(text: str):
        """Escape the handful of characters ReportLab treats as markup."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )

    # Orchestrator
    def run(self, fmt: str):
        """
        Execute fetch → extract → save.

        fmt: "txt" or "pdf".
        Returns the path to the written file.
        """
        if fmt not in ("txt", "pdf"):
            raise ValueError(f"Unsupported format: {fmt!r} (use 'txt' or 'pdf').")

        self.dest_dir.mkdir(parents=True, exist_ok=True)

        html = self.fetch()
        self.extract(html)
        out = self.save_txt() if fmt == "txt" else self.save_pdf()

        logging.info(f"Saved {fmt.upper()} → {out}  ({out.stat().st_size:,} bytes)")
        return out



# CLI entry point — `python -m utils.doc_parser <url> <dest_dir> --txt`

def main():
    parser = argparse.ArgumentParser(
        description="Fetch a web page and save its text as a .txt or .pdf "
                    "document for the RAG knowledge base.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python -m utils.doc_parser "
            "https://en.wikipedia.org/wiki/Income_inequality_in_the_United_States "
            "data/documents --txt\n"
            "  python -m utils.doc_parser "
            "https://archive.ics.uci.edu/dataset/2/adult data/documents --pdf"
        )
    )
    parser.add_argument("url", help="URL of the page to download.")
    parser.add_argument("dest_dir", help="Destination directory (e.g. data/documents).")
    fmt_group = parser.add_mutually_exclusive_group(required=True)
    fmt_group.add_argument("--txt", action="store_true", help="Save the document as plain text.")
    fmt_group.add_argument("--pdf", action="store_true", help="Save the document as a PDF.")
    args = parser.parse_args()

    fmt = "txt" if args.txt else "pdf"
    try:
        DocumentParser(args.url, args.dest_dir).run(fmt)
    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
