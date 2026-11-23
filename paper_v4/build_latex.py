"""Build the KAIS LaTeX version (Springer Nature sn-jnl class, sn-basic numbered references) from the assembled
markdown (manuscript_v4_pre.md, supplement_v4_pre.md). Tables and figures keep the order checked by assemble_v4.py;
LaTeX numbers them by appearance, which equals the order of first citation. Output: latex/{main,esm1}.tex/.pdf"""
import re, os, subprocess, shutil, sys
from refs_springer import REFS
D = os.path.dirname(os.path.abspath(__file__)); L = f'{D}/latex'; os.makedirs(L, exist_ok=True)
TPL = sys.argv[1] if len(sys.argv) > 1 else None
for f in ['sn-jnl.cls', 'sn-basic.bst']:
    if TPL: shutil.copy(f'{TPL}/{f}', L)
os.makedirs(f'{L}/figures', exist_ok=True)
for f in os.listdir(f'{D}/figures'):
    if f.endswith('.pdf'): shutil.copy(f'{D}/figures/{f}', f'{L}/figures/{f}')

UNI = {'−': '\\ensuremath{-}', '≤': '\\ensuremath{\\le}', '≥': '\\ensuremath{\\ge}', '≈': '\\ensuremath{\\approx}', '∧': '\\ensuremath{\\wedge}',
       'Σ': '\\ensuremath{\\Sigma}', 'λ': '\\ensuremath{\\lambda}', 'ε': '\\ensuremath{\\varepsilon}', 'ϵ': '\\ensuremath{\\epsilon}', 'θ': '\\ensuremath{\\theta}',
       'Δ': '\\ensuremath{\\Delta}', 'ρ': '\\ensuremath{\\rho}', 'π': '\\ensuremath{\\pi}', 'κ': '\\ensuremath{\\kappa}', 'μ': '\\ensuremath{\\mu}',
       'γ': '\\ensuremath{\\gamma}', 'η': '\\ensuremath{\\eta}', 'ϱ': '\\ensuremath{\\varrho}', 'φ': '\\ensuremath{\\phi}', '±': '\\ensuremath{\\pm}',
       '×': '\\ensuremath{\\times}', '→': '\\ensuremath{\\to}', '√': '\\ensuremath{\\surd}', '·': '\\ensuremath{\\cdot}', '¬': '\\ensuremath{\\neg}',
       '\u00a0': '~', '\u2009': '\\,', '\u2011': '-', 'ė': '\\.{e}', '✓': '\\ensuremath{\\checkmark}', '…': '\\ldots{}', '¹': '\\textsuperscript{1}', 'α': '\\ensuremath{\\alpha}', 'β': '\\ensuremath{\\beta}', 'δ': '\\ensuremath{\\delta}', 'τ': '\\ensuremath{\\tau}', 'Ω': '\\ensuremath{\\Omega}', 'Ψ': '\\ensuremath{\\Psi}', 'σ': '\\ensuremath{\\sigma}'}

TW, CH, CB, SEP = 372.0, 4.0, 4.6, 6.0     # text width (pt) and footnotesize glyph widths measured with sn-jnl
WIDTHS = []
def _cw(x, bold):
    if x.isupper(): return 6.5 if bold else 5.6
    if x.isdigit(): return 4.6 if bold else 4.0
    return 4.3 if bold else 3.8
def _w(t, head=False):
    """(natural width, longest unbreakable token) of a cell in pt; only ordinary spaces break a line."""
    bold = '**' in t or head
    t = re.sub(r'\$[^$]*\$', lambda m: 'x' * max(2, int(len(m.group(0)) * 0.45)), t).replace('**', '').replace('*', '')
    ln = lambda u: sum(_cw(x, bold) for x in u)
    tok = 1.08 * max((ln(x) for x in t.split(' ')), default=4.0)
    return (tok if head else 1.08 * ln(t)), tok
def cells(line):
    """Split a pipe-table row, ignoring pipes inside $...$."""
    out, cur, math = [], '', False
    for ch in line.strip().strip('|'):
        if ch == '$': math = not math
        if ch == '|' and not math: out.append(cur.strip()); cur = ''
        else: cur += ch
    return out + [cur.strip()]
def colwidths(md):
    """Record, for each pipe table, absolute column widths: natural widths when the table fits the text width,
    otherwise natural widths shrunk towards the longest unbreakable token of each column."""
    out, lines, i = [], md.split('\n'), 0
    while i < len(lines):
        if lines[i].startswith('|') and i + 1 < len(lines) and re.match(r'^\|(\s*:?-+:?\s*\|)+\s*$', lines[i + 1]):
            j = i + 2
            while j < len(lines) and lines[j].startswith('|'): j += 1
            rows = [cells(l) for l in [lines[i]] + lines[i + 2:j]]
            n = len(rows[0]); nat = [0.0] * n; mn = [0.0] * n
            for r_i, r in enumerate(rows):
                for c in range(min(n, len(r))):
                    a, b = _w(r[c], head=(r_i == 0)); nat[c] = max(nat[c], a, b); mn[c] = max(mn[c], b)
            small = sum(mn) + SEP * n > TW
            if small: nat = [x * 0.86 for x in nat]; mn = [x * 0.86 for x in mn]
            avail = TW - (4.0 if small else SEP) * n
            if sum(nat) <= avail: wd = nat
            elif sum(mn) >= avail: wd = [m * avail / sum(mn) for m in mn]
            else:   # columns needing little extra width are completed first; the rest share the remainder by slack
                wd = list(mn); ex = avail - sum(mn); slack = [a - b for a, b in zip(nat, mn)]
                for c in sorted(range(n), key=lambda c: slack[c]):
                    if slack[c] <= 0.25 * ex: wd[c] += slack[c]; ex -= slack[c]; slack[c] = 0
                    else: break
                tot = sum(slack)
                if tot > 0: wd = [w_ + ex * s_ / tot for w_, s_ in zip(wd, slack)]
            WIDTHS.append(([round(x, 1) for x in wd], small))
            out += [lines[i], '|' + '|'.join('-' * 3 for _ in range(n)) + '|'] + lines[i + 2:j]; i = j
        else: out.append(lines[i]); i += 1
    return '\n'.join(out)

def convert(md, supp=False):
    md = re.sub(r'^---\ntitle:.*?\n---\n', '', md, flags=re.S)
    keys = []
    def cite(m):
        ks = re.findall(r'@(\w+)', m.group(1))
        for k in ks:
            if k not in keys: keys.append(k)
        return '\\cite{' + ','.join(ks) + '}'
    md = re.sub(r'\[(@[\w]+(?:;\s*@[\w]+)*)\]', cite, md)
    # headings: drop the manual numbers (the class numbers sections)
    md = re.sub(r'^(#+) (S?\d+(\.\d+)*\.?) ', r'\1 ', md, flags=re.M)
    # table captions: "**Table N** text" paragraph followed by a pipe table -> pandoc caption after the table
    def tcap(m):
        cap, table = m.group(2).strip(), m.group(3).rstrip('\n')
        return f'{table}\n\n: {cap}\n'
    md = re.sub(r'\*\*Table (S?\d+)\*\* ([^\n]*)\n\n((?:\|[^\n]*\n)+)', tcap, md)
    # figures
    md = re.sub(r'!\[\*\*Fig\. S?\d+\*\* (.*?)\]\(figures/(\w+)\.png\)', r'![\1](figures/\2.pdf){width=100%}', md)
    md = md.replace('\u00a0±\u00a0', ' ±\u00a0').replace('\u00a0/\u00a0', ' /\u00a0')   # allow a line break before ± and /
    WIDTHS.clear(); md = colwidths(md)
    return md, keys

def to_tex(md, name):
    global WIDTHS
    open(f'{L}/{name}_body.md', 'w').write(md)
    subprocess.run(['pandoc', f'{L}/{name}_body.md', '-f', 'markdown+tex_math_dollars+pipe_tables+implicit_figures+raw_tex',
                    '-t', 'latex', '--columns=50', '--wrap=none', '-o', f'{L}/{name}_body.tex'], check=True)
    t = open(f'{L}/{name}_body.tex').read()
    for k, v in UNI.items(): t = t.replace(k, v)
    k = [0]
    def spec(m):
        wd, small = WIDTHS[k[0]]; k[0] += 1
        return ('\\begingroup\\fontsize{6}{7.2}\\selectfont\\setlength{\\tabcolsep}{2pt}' if small else '\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}') + '\\begin{longtable}[]{@{}' + ''.join('>{\\raggedright\\arraybackslash}p{%.1fpt}' % x for x in wd) + '@{}}'
    t = re.sub(r'\\begin\{longtable\}\[\]\{@\{\}.*?@\{\}\}', spec, t, flags=re.S)
    assert k[0] == len(WIDTHS), (k[0], len(WIDTHS))
    t = t.replace('\\begin{minipage}[b]{\\linewidth}\\raggedright\n', '\\begin{minipage}[b]{\\linewidth}\\raggedright\\hspace{0pt}').replace('\\end{longtable}', '\\end{longtable}\\endgroup')
    t = re.sub(r'\\section\{Statements and Declarations\}', r'\\backmatter\n\\section*{Statements and Declarations}', t)
    t = re.sub(r'\\section\{Abstract\}.*?(?=\\section)', '', t, flags=re.S)
    return t

def bib(keys):
    def esc(s):
        urls = re.findall(r'https?://\S+', s); s = re.sub(r'https?://\S+', '@@URL@@', s)
        s = s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
        for k, v in UNI.items(): s = s.replace(k, v)
        for u in urls: s = s.replace('@@URL@@', '\\url{' + u.rstrip('.') + '}', 1)
        return s
    return '\\begin{thebibliography}{99}\n' + '\n'.join(f'\\bibitem{{{k}}} {esc(REFS[k])}' for k in keys) + '\n\\end{thebibliography}\n'

PRE = r"""\documentclass[pdflatex,sn-basic,Numbered,lineno]{sn-jnl}
\usepackage{graphicx}\usepackage{amsmath,amssymb,amsfonts}\usepackage{booktabs}\usepackage{longtable}\usepackage{array}\usepackage{calc}
\usepackage{xcolor}\usepackage{textcomp}\usepackage{url}\usepackage{amsthm}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\makeatletter\def\fps@figure{htbp}\makeatother
\makeatletter\setkeys{Gin}{keepaspectratio}\makeatother
\raggedbottom
"""
def front(md):
    ab = re.search(r'# Abstract\n\n(.*?)\n\n\*\*Keywords:\*\* (.*?)\n', md, flags=re.S)
    abstract = subprocess.run(['pandoc', '-f', 'markdown+tex_math_dollars', '-t', 'latex'], input=ab.group(1), capture_output=True, text=True).stdout.strip()
    for k, v in UNI.items(): abstract = abstract.replace(k, v)
    kw = ', '.join(x.strip() for x in ab.group(2).split(';'))
    title = re.search(r'title: "(.*?)"', md).group(1)
    return title, abstract, kw

# ---- main
main_md = open(f'{D}/manuscript_v4_pre.md').read()
title, abstract, kw = front(main_md)
body_md = main_md[main_md.index('# 1. Introduction'):]
body_md, keys = convert(body_md)
body = to_tex(body_md, 'main')
tex = PRE + r"""\begin{document}
\title[Fuzzy demographic parity and rule-local fairness constraints]{""" + title.replace('–', '--') + r"""}
\author*{\fnm{Rahib} \sur{Imamguluyev}}\email{rahib.aydinoglu@gmail.com}
\affil*{\orgdiv{Department of Information Technologies}, \orgname{Baku Business University}, \orgaddress{\street{88a H. Zardabi St.}, \city{Baku}, \postcode{AZ1122}, \country{Azerbaijan}}}
\abstract{""" + abstract + r"""}
\keywords{""" + kw + r"""}
\maketitle
""" + body + '\n' + bib(keys) + '\\end{document}\n'
open(f'{L}/main.tex', 'w').write(tex)
# ---- supplement (Online Resource 1)
s_md = open(f'{D}/supplement_v4_pre.md').read()
stitle = re.search(r'title: "(.*?)"', s_md).group(1)
s_body = s_md[s_md.index('# S1.'):]
intro = s_md[s_md.index('---', 3) + 3:s_md.index('# S1.')].strip()
intro = intro[intro.index('Section, equation'):]          # title block and abstract already carry journal, author and address
s_body, skeys = convert(intro + '\n\n' + s_body, supp=True)
sb = to_tex(s_body, 'esm1')
stex = PRE.replace(',lineno', '') + r"""\renewcommand{\thetable}{S\arabic{table}}\renewcommand{\thefigure}{S\arabic{figure}}\renewcommand{\thesection}{S\arabic{section}}
\begin{document}
\title{""" + stitle.replace('–', '--') + r"""}
\author*{\fnm{Rahib} \sur{Imamguluyev}}\email{rahib.aydinoglu@gmail.com}
\affil*{\orgdiv{Department of Information Technologies}, \orgname{Baku Business University}, \orgaddress{\street{88a H. Zardabi St.}, \city{Baku}, \postcode{AZ1122}, \country{Azerbaijan}}}
\abstract{Online Resource 1 to the article submitted to \emph{Knowledge and Information Systems}: proofs of auxiliary results, additional tables and robustness analyses.}
\maketitle
""" + sb + '\n' + bib(skeys) + '\\end{document}\n'
open(f'{L}/esm1.tex', 'w').write(stex)
for n in ['main', 'esm1']:
    for _ in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', f'{n}.tex'], cwd=L, capture_output=True, text=True)
    errs = [l for l in r.stdout.split('\n') if l.startswith('!')]
    und = re.findall(r'(?:Reference|Citation) `[^\']*\' .*?undefined', r.stdout)
    print(n, 'errors:', errs[:8], 'undefined:', und[:5])
    over = re.findall(r'Overfull \\hbox \((\d+\.\d+)pt', r.stdout); print(n, 'overfull >10pt:', [o for o in over if float(o) > 10][:10])
