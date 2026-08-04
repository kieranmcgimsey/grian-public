"""Build the grian learning guide as a multi-chapter PDF.

Usage:
    python build.py              # build all guides + merged PDF
    python build.py 01           # build just guide 01
"""
import base64
import io
import re
import sys
from pathlib import Path

import markdown
from xhtml2pdf import pisa

GUIDE_DIR = Path(__file__).parent
SRC_DIR = GUIDE_DIR / "src"
PDF_DIR = GUIDE_DIR / "pdf"
FIG_DIR = GUIDE_DIR / "figures"

CSS = """
@page {
    size: A4;
    margin: 1.5cm;
    @frame footer {
        -pdf-frame-content: footerFrame;
        bottom: 0.3cm;
        margin-left: 1.5cm;
        margin-right: 1.5cm;
        height: 1cm;
    }
}
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 8pt;
    line-height: 1.3;
    color: #1a1a1a;
}
h1 {
    font-size: 14pt;
    color: #1a3a5c;
    border-bottom: 2px solid #2c5f8a;
    padding-bottom: 4pt;
    margin-top: 16pt;
}
h2 {
    font-size: 11pt;
    color: #2c5f8a;
    margin-top: 12pt;
}
h3 { font-size: 9.5pt; color: #3a7ab5; margin-top: 8pt; }
h4 { font-size: 8pt; color: #555555; font-style: italic; }
p { margin: 3pt 0; }
a { color: #2c5f8a; text-decoration: none; }
code {
    background-color: #f4f4f4;
    padding: 1px 3px;
    font-size: 7pt;
}
pre {
    background-color: #f8f8f8;
    padding: 6pt;
    font-size: 6.5pt;
    line-height: 1.3;
    border: 1px solid #e0e0e0;
    margin: 4pt 0;
    white-space: pre-wrap;
    word-wrap: break-word;
}
pre code { background-color: transparent; padding: 0; }
.key-point {
    padding: 6pt 8pt;
    margin: 6pt 0;
    background-color: #f0f5fa;
    color: #333333;
    border-left: 3px solid #2c5f8a;
    font-size: 7.5pt;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 5pt 0;
    font-size: 7.5pt;
}
th, td {
    border: 1px solid #cccccc;
    padding: 2px 4px;
    text-align: left;
}
th { background-color: #e8eff5; font-weight: bold; }
ul, ol { margin: 2pt 0; padding-left: 16pt; }
li { margin: 1pt 0; }
hr { border: none; border-top: 1px solid #dddddd; margin: 8pt 0; }
img {
    max-width: 100%;
    display: block;
    margin: 6pt auto;
}
.figure-caption {
    text-align: center;
    font-size: 7pt;
    color: #666666;
    font-style: italic;
    margin-top: 2pt;
}
.definition-box {
    padding: 6pt 8pt;
    margin: 6pt 0;
    background-color: #e8f4fd;
    border-left: 3px solid #3498db;
    font-size: 7.5pt;
}
.definition-box strong { color: #1a3a5c; }
.example-box {
    background-color: #fdf6e3;
    padding: 6pt 8pt;
    margin: 6pt 0;
    border-left: 3px solid #d4a017;
    font-size: 7.5pt;
}
.exercise-box {
    background-color: #eafaf1;
    padding: 6pt 8pt;
    margin: 6pt 0;
    border-left: 3px solid #27ae60;
    font-size: 7.5pt;
}
.toc-chapter {
    font-size: 11pt;
    font-weight: bold;
    margin: 6pt 0 2pt 0;
}
.toc-chapter a {
    color: #1a3a5c;
    text-decoration: none;
}
.toc-section {
    font-size: 10pt;
    margin-left: 24pt;
    color: #2c5f8a;
}
.toc-page {
    color: #888888;
    font-size: 7pt;
}
"""

MD_EXTENSIONS = ["tables", "fenced_code", "attr_list", "md_in_html"]

CHAPTERS = [
    ("01", "Data Ingestion and the NEM", "01_data_ingestion"),
    ("02", "Price Distribution and Stylised Facts", "02_price_distribution"),
    ("03", "Price Formation", "03_price_formation"),
    ("04", "Weather and Renewables", "04_weather_and_renewables"),
    ("05", "Forecasting Framework and Baselines", "05_framing_and_baselines"),
    ("06", "Classical Models: LEAR", "06_classical_models"),
    ("07", "Machine Learning", "07_machine_learning"),
    ("08", "Probabilistic Forecasting", "08_probabilistic_forecasting"),
    ("09", "From Forecast to Money", "09_forecast_to_money"),
    ("10", "Capstone: Sunlight to Revenue", "10_capstone"),
    ("11", "FCAS and Co-optimised Dispatch", "11_fcas_and_cooptimisation"),
    ("12", "Bidding and Market Participation", "12_bidding_and_participation"),
    ("13", "Realised Revenue", "13_realised_revenue"),
    ("14", "Failure Analysis and Explainability", "14_failure_analysis"),
    ("15", "Live Operations and the Intra-day Loop", "15_live_operations"),
    ("16", "Decision-Focused Learning", "16_decision_focused_learning"),
]


def _apply_sub_sup(html: str) -> str:
    """Convert notation to <sub>/<sup> in text nodes only."""
    parts = re.split(r'(<[^>]+>)', html)
    skip_depth = 0
    skip_tags = {'code', 'pre', 'style'}
    result = []

    for part in parts:
        if part.startswith('<'):
            tag_match = re.match(r'<(/?)(\w+)', part)
            if tag_match:
                tag_name = tag_match.group(2).lower()
                if tag_name in skip_tags:
                    if tag_match.group(1) == '/':
                        skip_depth = max(0, skip_depth - 1)
                    else:
                        skip_depth += 1
            result.append(part)
            continue

        if skip_depth > 0:
            result.append(part)
            continue

        # Braced forms first
        part = re.sub(r'_\{([^}]+)\}', r'<sub>\1</sub>', part)
        part = re.sub(r'\^\{([^}]+)\}', r'<sup>\1</sup>', part)

        # Unbraced digit subscripts
        part = re.sub(
            r'(?<=[A-Za-zͰ-Ͽ*])_(\d+(?:\.\d+)?)',
            r'<sub>\1</sub>', part,
        )
        # Single letter subscripts (not multi-letter words)
        part = re.sub(
            r'(?<=[A-Za-zͰ-Ͽ*])_([a-z])(?![a-z_\d])',
            r'<sub>\1</sub>', part,
        )

        # Unbraced superscripts - digits
        part = re.sub(
            r'(?<=[A-Za-zͰ-Ͽ\d\)*])\^(-?\d+)',
            r'<sup>\1</sup>', part,
        )
        # Single letter superscripts
        part = re.sub(
            r'(?<=[A-Za-zͰ-Ͽ\d\)*])\^([a-zA-Z])(?![a-zA-Z])',
            r'<sup>\1</sup>', part,
        )

        result.append(part)

    return ''.join(result)


def _fix_table_cells(html: str) -> str:
    """Fill empty table cells so xhtml2pdf does not collapse their columns."""
    def _fix_table(match):
        return match.group(0).replace('<td></td>', '<td>-</td>')

    return re.sub(r'<table[^>]*>.*?</table>', _fix_table, html, flags=re.DOTALL)


def _fix_details_blocks(text: str) -> str:
    """Add markdown='1' to <details> so md_in_html processes inner markdown."""
    text = text.replace('<details>', '<details markdown="1">')
    text = text.replace('<summary>', '<summary markdown="1">')
    return text


def _embed_images(html: str) -> str:
    """Replace img src with base64 data URIs for xhtml2pdf."""
    def _replace(match):
        src = match.group(1)
        img_path = FIG_DIR / Path(src).name if "figures/" in src else SRC_DIR / src
        if not img_path.exists():
            img_path = GUIDE_DIR / src
        if img_path.exists():
            data = base64.b64encode(img_path.read_bytes()).decode()
            return f'src="data:image/png;base64,{data}"'
        return match.group(0)
    return re.sub(r'src="([^"]+)"', _replace, html)


def _slugify(text: str) -> str:
    """Convert heading text to URL-safe anchor slug."""
    text = re.sub(r'<[^>]+>', '', text)
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s]+', '-', text)
    return text


def _extract_headings(md_text: str) -> list:
    """Extract h2 and h3 headings from markdown text."""
    headings = []
    for line in md_text.split("\n"):
        if line.startswith("### "):
            headings.append((3, line[4:].strip()))
        elif line.startswith("## "):
            headings.append((2, line[3:].strip()))
    return headings


def _add_heading_anchors(html: str, chapter_prefix: str) -> str:
    """Add id attributes to h1/h2/h3 tags for internal linking."""
    def _add_id(match):
        tag = match.group(1)
        attrs = match.group(2) or ""
        content = match.group(3)
        slug = f"{chapter_prefix}-{_slugify(content)}"
        anchor = f'<a name="{slug}"></a>'
        return f'{anchor}<{tag}{attrs}>{content}</{tag}>'

    html = re.sub(
        r'<(h[123])(\s[^>]*)?>(.+?)</\1>',
        _add_id,
        html,
    )
    return html


def _build_cover_html() -> str:
    """Generate cover page HTML fragment."""
    return """
<p style="font-size: 14pt; letter-spacing: 4pt; color: #2c5f8a; margin-top: 180pt; margin-bottom: 20pt;">LEARNING GUIDE</p>
<p style="font-size: 42pt; font-weight: bold; line-height: 1.2; margin-bottom: 10pt; color: #1a3a5c;">grian</p>
<p style="font-size: 18pt; color: #3a7ab5; margin-bottom: 40pt;">NEM Electricity Price Forecasting<br/>and Battery Dispatch</p>
<hr style="border: none; border-top: 3px solid #1a3a5c; width: 80pt; margin: 0 0 30pt 0;"/>
<p style="font-size: 12pt; color: #555; line-height: 1.8;">
    From raw market data to probabilistic forecasts<br/>
    and forecast-driven battery revenue optimisation
</p>
<p style="font-size: 10pt; color: #888; margin-top: 200pt;">
    10 chapters &bull; 24 figures &bull; 200+ definitions &bull; zero assumed knowledge
</p>
"""


def _build_toc_html(page_numbers: dict | None = None) -> str:
    """Generate table of contents HTML with hyperlinks and optional page numbers."""
    toc_parts = ['<h1 style="border-bottom: 3px solid #1a3a5c; padding-bottom: 8pt; margin-top: 0pt;">Contents</h1>']

    for ch_num, ch_title, ch_file in CHAPTERS:
        ch_prefix = ch_num
        ch_anchor = f"ch{ch_num}"
        page_str = ""
        if page_numbers and ch_anchor in page_numbers:
            page_str = f' <span class="toc-page">... {page_numbers[ch_anchor]}</span>'
        toc_parts.append(
            f'<p class="toc-chapter"><a href="#{ch_anchor}">{ch_num}. {ch_title}</a>{page_str}</p>'
        )

        md_path = SRC_DIR / f"{ch_file}.md"
        if md_path.exists():
            headings = _extract_headings(md_path.read_text())
            for level, heading in headings:
                if heading.lower() in ("glossary", "summary"):
                    continue
                slug = f"{ch_prefix}-{_slugify(heading)}"
                page_str = ""
                if page_numbers and slug in page_numbers:
                    page_str = f' <span class="toc-page">... {page_numbers[slug]}</span>'
                indent = "24pt" if level == 2 else "48pt"
                toc_parts.append(
                    f'<p style="margin: 1pt 0 1pt {indent}; text-align: left;">'
                    f'<a href="#{slug}" style="font-size: {"10pt" if level == 2 else "9pt"}; '
                    f'color: {"#2c5f8a" if level == 2 else "#666"}; text-decoration: none;">'
                    f'{heading}</a>{page_str}</p>'
                )

    toc_parts.append("""
<div style="margin-top: 30pt; padding: 14pt; background-color: #f0f5fa; border-left: 4px solid #2c5f8a;">
    <p style="font-size: 10pt; color: #333; margin: 0; text-align: left;"><strong>How to use this guide:</strong> Each chapter corresponds to a hands-on notebook. Read the chapter first to understand the concepts, then work through the notebook to apply them. Every technical term is defined in blue definition boxes. Real-world examples appear in orange boxes. Key insights are highlighted in green boxes. Each chapter ends with a glossary of all terms introduced.</p>
</div>
""")

    return "\n".join(toc_parts)


def _build_glossary_html() -> str:
    """Build master glossary HTML from all chapter glossary tables."""
    all_terms = {}

    for md_path in sorted(SRC_DIR.glob("*.md")):
        text = md_path.read_text()
        chapter_num = md_path.name.split("_")[0]

        in_glossary = False
        for line in text.split("\n"):
            if line.strip().startswith("## Glossary"):
                in_glossary = True
                continue
            if in_glossary and line.strip().startswith("## "):
                in_glossary = False
                continue
            if in_glossary and line.startswith("|") and "**" in line:
                parts = [p.strip() for p in line.split("|")[1:-1]]
                if len(parts) == 2:
                    term = parts[0].replace("**", "").strip()
                    defn = parts[1].strip()
                    if term.lower() not in ("term", "---", ""):
                        all_terms[term] = (defn, chapter_num)

    sorted_terms = sorted(all_terms.items(), key=lambda x: x[0].lower())

    rows = ""
    current_letter = ""
    for term, (defn, ch) in sorted_terms:
        first = term[0].upper()
        if first != current_letter:
            current_letter = first
            rows += f"""
            <tr>
                <td colspan="3" style="border: none; border-bottom: 2px solid #2c5f8a; padding: 12pt 4pt 2pt 4pt;">
                    <strong style="font-size: 14pt; color: #2c5f8a;">{current_letter}</strong>
                </td>
            </tr>"""
        rows += f"""
        <tr>
            <td style="font-weight: bold; color: #1a3a5c; width: 25%; vertical-align: top;">{term}</td>
            <td style="width: 65%;">{defn}</td>
            <td style="width: 10%; text-align: center; color: #888; font-size: 9pt;">Ch {ch}</td>
        </tr>"""

    rows = _apply_sub_sup(rows)
    return f"""
<h1 style="margin-top: 0pt;"><a name="glossary"></a>Master Glossary</h1>
<p style="color: #666; font-size: 10pt;">All terms defined across all chapters, listed alphabetically with their source chapter.</p>
<table style="font-size: 9.5pt;">
<tr><th style="width: 25%;">Term</th><th style="width: 65%;">Definition</th><th style="width: 10%;">Chapter</th></tr>
{rows}
</table>
"""


def _build_full_html(page_numbers: dict | None = None) -> str:
    """Build the complete guide as a single HTML document."""
    body_parts = []

    # Cover page
    body_parts.append(_build_cover_html())

    # TOC
    body_parts.append('<div style="page-break-before: always;">')
    body_parts.append(_build_toc_html(page_numbers))
    body_parts.append('</div>')

    # Chapters
    for ch_num, ch_title, ch_file in CHAPTERS:
        md_path = SRC_DIR / f"{ch_file}.md"
        if not md_path.exists():
            print(f"  WARNING: {md_path} not found, skipping")
            continue

        text = md_path.read_text()
        text = _fix_details_blocks(text)
        html_body = markdown.markdown(text, extensions=MD_EXTENSIONS)
        html_body = _apply_sub_sup(html_body)
        html_body = _fix_table_cells(html_body)
        html_body = _embed_images(html_body)
        html_body = _add_heading_anchors(html_body, ch_num)

        # Add chapter-level anchor to the first h1
        ch_anchor = f"ch{ch_num}"
        html_body = html_body.replace(
            f'<a name="{ch_num}-',
            f'<a name="{ch_anchor}"></a><a name="{ch_num}-',
            1,
        )
        # Fix: if first heading already got double anchor, clean up
        # Actually, let's just prepend the chapter anchor before the first heading
        html_body = f'<a name="{ch_anchor}"></a>\n{html_body}'

        body_parts.append('<div style="page-break-before: always;">')
        body_parts.append(html_body)
        body_parts.append('</div>')

    # Master glossary
    body_parts.append('<div style="page-break-before: always;">')
    body_parts.append(_build_glossary_html())
    body_parts.append('</div>')

    body_content = "\n".join(body_parts)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{body_content}
<div id="footerFrame">
    <p style="text-align: center; font-size: 9pt; color: #666666;">
        <pdf:pagenumber/> | grian learning guide
    </p>
</div>
</body></html>"""


def _extract_page_numbers(pdf_bytes: bytes) -> dict:
    """Extract page numbers by following link annotation destinations."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))

    # Build page lookup by indirect reference idnum
    page_idnums = {}
    for i, page in enumerate(reader.pages):
        ref = page.indirect_reference
        if ref:
            page_idnums[ref.idnum] = i + 1

    # Build ordered list of all TOC link anchors (chapters + sections)
    toc_anchors = []
    for ch_num, ch_title, ch_file in CHAPTERS:
        toc_anchors.append(f"ch{ch_num}")
        md_path = SRC_DIR / f"{ch_file}.md"
        if md_path.exists():
            for level, heading in _extract_headings(md_path.read_text()):
                if heading.lower() in ("glossary", "summary"):
                    continue
                toc_anchors.append(f"{ch_num}-{_slugify(heading)}")

    # Collect link destinations from all pages in order
    all_link_dests = []
    for page in reader.pages:
        if "/Annots" in page:
            annots = page["/Annots"]
            for annot in annots:
                annot_obj = annot.get_object()
                if "/Dest" in annot_obj:
                    dest = annot_obj["/Dest"]
                    if isinstance(dest, list) and len(dest) >= 1:
                        page_ref = dest[0]
                        page_obj = page_ref.get_object() if hasattr(page_ref, 'get_object') else page_ref
                        page_ref_id = page_ref.idnum if hasattr(page_ref, 'idnum') else None
                        if page_ref_id and page_ref_id in page_idnums:
                            all_link_dests.append(page_idnums[page_ref_id])

    # Match destinations to TOC anchors in order
    page_numbers = {}
    anchor_idx = 0
    for dest_page in all_link_dests:
        if anchor_idx < len(toc_anchors):
            page_numbers[toc_anchors[anchor_idx]] = dest_page
            anchor_idx += 1

    return page_numbers


def _build_single_chapter(ch_num: str, ch_title: str, ch_file: str) -> None:
    """Build a single chapter as a standalone PDF."""
    md_path = SRC_DIR / f"{ch_file}.md"
    if not md_path.exists():
        print(f"  WARNING: {md_path} not found, skipping")
        return

    text = md_path.read_text()
    text = _fix_details_blocks(text)
    html_body = markdown.markdown(text, extensions=MD_EXTENSIONS)
    html_body = _apply_sub_sup(html_body)
    html_body = _fix_table_cells(html_body)
    html_body = _embed_images(html_body)

    full_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{html_body}
<div id="footerFrame">
    <p style="text-align: center; font-size: 9pt; color: #666666;">
        <pdf:pagenumber/> | grian learning guide
    </p>
</div>
</body></html>"""

    PDF_DIR.mkdir(exist_ok=True)
    out_path = PDF_DIR / f"{ch_file}.pdf"
    with open(out_path, "wb") as f:
        pisa.CreatePDF(full_html, dest=f)

    size_kb = out_path.stat().st_size // 1024
    print(f"  {ch_file}.pdf ({size_kb} KB)")


def main():
    """Build individual chapter PDFs and the merged guide."""
    PDF_DIR.mkdir(exist_ok=True)

    # Check if building a single chapter
    if len(sys.argv) > 1:
        target = sys.argv[1]
        for ch_num, ch_title, ch_file in CHAPTERS:
            if ch_num == target or ch_file.startswith(target):
                _build_single_chapter(ch_num, ch_title, ch_file)
                return
        print(f"Chapter '{target}' not found")
        return

    # Build individual chapter PDFs
    print("Building individual chapter PDFs...")
    for ch_num, ch_title, ch_file in CHAPTERS:
        _build_single_chapter(ch_num, ch_title, ch_file)

    # Build merged guide
    print("\nBuilding merged guide...")

    # Pass 1: generate PDF without page numbers to discover layout
    print("Pass 1: generating layout...")
    html1 = _build_full_html()
    buf1 = io.BytesIO()
    pisa.CreatePDF(html1, dest=buf1)

    # Extract page numbers from pass 1
    page_numbers = _extract_page_numbers(buf1.getvalue())
    for slug, page in sorted(page_numbers.items()):
        print(f"  {slug}: page {page}")

    # Pass 2: rebuild with page numbers in TOC
    print("Pass 2: rebuilding with page numbers in TOC...")
    html2 = _build_full_html(page_numbers)
    out_path = GUIDE_DIR / "grian_learning_guide.pdf"
    with open(out_path, "wb") as f:
        pisa.CreatePDF(html2, dest=f)

    size_kb = out_path.stat().st_size // 1024
    from pypdf import PdfReader
    reader = PdfReader(str(out_path))
    print(f"\nComplete: {out_path.name} ({size_kb} KB, {len(reader.pages)} pages)")


if __name__ == "__main__":
    main()
