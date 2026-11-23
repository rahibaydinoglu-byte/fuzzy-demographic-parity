"""manuscript_v3.md -> manuscript_v3.docx (pandoc with native Word equations), then styling with python-docx:
Times New Roman 12 pt, 1.5 line spacing, continuous line numbers, page numbers, booktabs-like tables, 9-pt tables."""
import os, subprocess, sys
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
D = os.path.dirname(os.path.abspath(__file__))
MD = os.environ.get('MD', f'{D}/manuscript_v3.md'); OUT = os.environ.get('OUT', f'{D}/manuscript_v3.docx')
subprocess.run(['pandoc', MD, '-f', 'markdown+tex_math_dollars+pipe_tables+implicit_figures', '-o', OUT,
                '--resource-path', D, '--wrap=none'], check=True, cwd=D)
doc = Document(OUT)
for st in doc.styles:
    try:
        if st.type == 1:
            st.font.name = 'Times New Roman'; st.font.color.rgb = RGBColor(0, 0, 0)
            rpr = st.element.get_or_add_rPr(); rf = rpr.find(qn('w:rFonts'))
            if rf is None: rf = OxmlElement('w:rFonts'); rpr.append(rf)
            for a in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'): rf.set(qn(a), 'Times New Roman')
    except Exception: pass
for name, size in [('Normal', 12), ('Body Text', 12), ('First Paragraph', 12), ('Compact', 12), ('Title', 16),
                   ('Heading 1', 13), ('Heading 2', 12), ('Heading 3', 12)]:
    cand = [x for x in doc.styles if x.name and x.name.lower() == name.lower()]
    if cand:
        s = cand[0]; s.font.size = Pt(size)
        if name.startswith('Heading') or name == 'Title': s.font.bold = True; s.font.italic = name == 'Heading 2' and False
        pf = s.paragraph_format; pf.line_spacing = 1.5
        if name.startswith('Heading'): pf.space_before = Pt(12); pf.space_after = Pt(6)
sec = doc.sections[0]
sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'): setattr(sec, side, Cm(2.5))
ln = OxmlElement('w:lnNumType'); ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '284')
sec._sectPr.append(ln)
# page number footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sl = OxmlElement('w:suppressLineNumbers'); fp._p.get_or_add_pPr().append(sl)   # avoid a second number from line numbering
for t, txt in [('begin', None), (None, 'PAGE'), ('end', None)]:
    r = fp.add_run()
    if t: fc = OxmlElement('w:fldChar'); fc.set(qn('w:fldCharType'), t); r._r.append(fc)
    else: it = OxmlElement('w:instrText'); it.set(qn('xml:space'), 'preserve'); it.text = txt; r._r.append(it)
# tables: 9 pt, single spacing, top/bottom rules + header rule, autofit to page width
def border(el, tag, val='single', sz='8'):
    b = OxmlElement(f'w:{tag}'); b.set(qn('w:val'), val); b.set(qn('w:sz'), sz); b.set(qn('w:space'), '0'); b.set(qn('w:color'), '000000'); el.append(b)
for tb in doc.tables:
    tbl = tb._tbl; tblPr = tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')): tblPr.remove(old)
    bd = OxmlElement('w:tblBorders'); border(bd, 'top', sz='12'); border(bd, 'bottom', sz='12')
    for t in ('left', 'right', 'insideH', 'insideV'): border(bd, t, val='nil')
    tblPr.append(bd)
    w = OxmlElement('w:tblW'); w.set(qn('w:w'), '5000'); w.set(qn('w:type'), 'pct')
    for old in tblPr.findall(qn('w:tblW')): tblPr.remove(old)
    tblPr.append(w)
    for i, row in enumerate(tb.rows):
        for cell in row.cells:
            if i == 0:
                tcPr = cell._tc.get_or_add_tcPr(); tb_ = OxmlElement('w:tcBorders'); border(tb_, 'bottom', sz='6'); tcPr.append(tb_)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing = 1.0; p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
                for r in p.runs: r.font.size = Pt(9)

# column widths proportional to content (LibreOffice and Word otherwise keep pandoc's equal widths)
TEXTW = 16.0
for tb in doc.tables:
    ncol = len(tb.columns); wide = ncol >= 8
    for row in tb.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for r in p.runs:
                    if wide: r.text = r.text.replace(' / ', '/').replace(' ± ', '±').replace('\u00a0±\u00a0', '±')
    lens = [10] * ncol
    longest = [max(len(r.cells[jj].text) for r in tb.rows) for jj in range(ncol)]
    for ri, row in enumerate(tb.rows):
        for jj, cell in enumerate(row.cells[:ncol]):
            t = cell.text
            L = max(len(w) for w in (t.split() or [' '])) if ri == 0 else len(t)
            lens[jj] = max(lens[jj], min(L, 42 if longest[jj] > 100 else 26))
    widths = [TEXTW * l / sum(lens) for l in lens]
    grid = tb._tbl.tblGrid
    for jj, gc in enumerate(grid.findall(qn('w:gridCol'))):
        if jj < ncol: gc.set(qn('w:w'), str(int(widths[jj] * 567)))
    for row in tb.rows:
        for jj, cell in enumerate(row.cells[:ncol]):
            cell.width = Cm(widths[jj])
    tblPr = tb._tbl.tblPr
    for old in tblPr.findall(qn('w:tblLayout')): tblPr.remove(old)
    lay = OxmlElement('w:tblLayout'); lay.set(qn('w:type'), 'fixed'); tblPr.append(lay)
    if ncol >= 8:
        for row in tb.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs: r.font.size = Pt(7)

# keep each table on one page where possible: rows not split, every row kept with the next, header repeated
for tb in doc.tables:
    for ri, row in enumerate(tb.rows):
        trPr = row._tr.get_or_add_trPr()
        cs = OxmlElement('w:cantSplit'); trPr.append(cs)
        if ri == 0: th = OxmlElement('w:tblHeader'); trPr.append(th)
        if ri < len(tb.rows) - 1:
            for cell in row.cells:
                for p in cell.paragraphs: p.paragraph_format.keep_with_next = True
# captions of tables/figures: keep with next, single spacing
for p in doc.paragraphs:
    t = p.text.strip()
    import re as _re
    if _re.match(r'^(Table S?\d+|Fig\. S?\d+)\.?\s', t):
        p.paragraph_format.keep_with_next = True; p.paragraph_format.line_spacing = 1.15
        for r in p.runs: r.font.size = Pt(10)
# images: width 16 cm
from docx.shared import Emu
for shp in doc.inline_shapes:
    ratio = shp.height / shp.width; shp.width = Cm(16); shp.height = int(Cm(16) * ratio)
doc.core_properties.author = 'Rahib Imamguluyev'; doc.core_properties.title = 'Fuzzy demographic parity, rule-level disparity decomposition and the failure of rule-local fairness constraints in Takagi–Sugeno–Kang classifiers'
doc.save(OUT); print('saved', OUT)
