"""Assemble manuscript v4 and its supplement. Tables/figures/references numbered by first citation; main-text
references to the supplement (@stbl:key, @sfig:key) resolved from the supplement's numbering; supplement references
to main tables (@mtbl:key) resolved from the main numbering. The model is called FairTSK throughout (ZadehNet in code)."""
import re, os
from refs_springer import REFS
D = os.path.dirname(os.path.abspath(__file__))
errors = []
def ins(md):
    def f(m):
        fn = f'{D}/tables/{m.group(1)}.md'
        if not os.path.exists(fn): errors.append(f'missing table file {m.group(1)}'); return m.group(0)
        t = open(fn).read().strip()
        if m.group(1) == 'continuous': t = re.sub(r'\s*±\s*\d\.\d+', '', t)   # s.d. given in the caption
        return t
    return re.sub(r'\{\{TABLE:(\w+)\}\}', f, md)
def rename(md):
    md = md.replace('ZadehNet-rule', 'FairTSK-rule').replace('ZadehNet-0', 'FairTSK-0').replace('ZadehNet0', 'FairTSK-0').replace('ZadehNet', 'FairTSK')
    return md.replace('@@ZADEHNET@@', 'ZadehNet')
def number(md, prefix, tag):
    order = []
    for m in re.finditer(r'(?<![sm])@tbl:(\w+)|\{#tbl:(\w+)\}', md):
        k = m.group(1) or m.group(2)
        if m.group(2) and k not in order and tag == 'main': errors.append(f'{tag} table {k}: caption before first citation')
        if k not in order: order.append(k)
    tn = {k: f'{prefix}{i + 1}' for i, k in enumerate(order)}
    for k in re.findall(r'\{#tbl:(\w+)\}', md):
        if k not in tn: errors.append(f'{tag} table {k} never cited')
    forder = []
    for m in re.finditer(r'(?<![sm])@fig:(\w+)|\{#fig:(\w+)\}', md):
        k = m.group(1) or m.group(2)
        if m.group(2) and k not in forder and tag == 'main': errors.append(f'{tag} figure {k}: before first citation')
        if k not in forder: forder.append(k)
    fn = {k: f'{prefix}{i + 1}' for i, k in enumerate(forder)}
    md = re.sub(r'(?<![sm])@tbl:(\w+)', lambda m: f'Table {tn.get(m.group(1), "??")}', md)
    md = re.sub(r'\*\*Table \{#tbl:(\w+)\}\.\*\*', lambda m: f'**Table {tn.get(m.group(1), "??")}**', md)
    md = re.sub(r'!\[(.*?)\]\((.*?)\)\{#fig:(\w+)\}', lambda m: f'![**Fig. {fn[m.group(3)]}** {m.group(1).rstrip(".")}]({m.group(2)})', md)
    md = re.sub(r'(?<![sm])@fig:(\w+)', lambda m: f'Fig. {fn.get(m.group(1), "??")}', md)
    return md, tn, fn
def cites(md, tag):
    rorder = []
    for grp in re.findall(r'\[(@[\w]+(?:;\s*@[\w]+)*)\]', md):
        for k in re.findall(r'@(\w+)', grp):
            if k not in REFS: errors.append(f'{tag}: unknown reference key {k}')
            elif k not in rorder: rorder.append(k)
    rn = {k: i + 1 for i, k in enumerate(rorder)}
    def c(m):
        ns = sorted(rn[k] for k in re.findall(r'@(\w+)', m.group(1)) if k in rn); out, i = [], 0
        while i < len(ns):
            j = i
            while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
            out.append(f'{ns[i]}–{ns[j]}' if j - i >= 2 else (f'{ns[i]},{ns[j]}' if j == i + 1 else f'{ns[i]}')); i = j + 1
        return '[' + ','.join(out) + ']'
    md = re.sub(r'\[(@[\w]+(?:;\s*@[\w]+)*)\]', c, md)
    return md + '\n\n# References\n\n' + '\n\n'.join(f'[{rn[k]}] {REFS[k]}' for k in rorder) + '\n', rorder
def checks(md, tag):
    for pat in [r'@\w+:', r'\{\{', r'\?\?', r'\[@', r'@@']:
        for m in re.finditer(pat, md): errors.append(f'{tag}: leftover {m.group(0)!r} near {md[max(0, m.start()-50):m.start()+30]!r}')
def tags(md, tag):
    eq = [int(x) for x in re.findall(r'\\tag\{(\d+)\}', md)]
    if eq != list(range(1, len(eq) + 1)): errors.append(f'{tag}: equation tags {eq}')
    return re.sub(r'\s*\\tag\{(\d+)\}\s*\$\$', lambda m: r'\qquad\qquad(' + m.group(1) + ')$$', md)
main = rename(ins('\n\n'.join(open(f'{D}/{p}').read() for p in ['q1_front.md', 'q2_theory.md', 'q3_experiments.md', 'q4_end.md'])))
supp = rename(ins(open(f'{D}/s_supp.md').read()))
supp, stn, sfn = number(supp, 'S', 'supp')
main = re.sub(r'@stbl:(\w+)', lambda m: f'Table {stn.get(m.group(1), "??")}', main)
main = re.sub(r'@sfig:(\w+)', lambda m: f'Fig. {sfn.get(m.group(1), "??")}', main)
main = main.replace('@@NSTAB@@', str(len(stn))).replace('@@NSFIG@@', str(len(sfn)))
main, tn, fn = number(main, '', 'main')
supp = re.sub(r'@mtbl:(\w+)', lambda m: f'Table {tn.get(m.group(1), "??")} of the main article', supp)
open(f'{D}/manuscript_v4_pre.md', 'w').write(main); open(f'{D}/supplement_v4_pre.md', 'w').write(supp)   # for the LaTeX build
main, rm = cites(main, 'main'); supp, rs = cites(supp, 'supp')
for k in REFS:
    if k not in rm and k not in rs: errors.append(f'reference {k} cited nowhere')
for name in ['Proposition', 'Theorem', 'Corollary', 'Lemma']:
    defs = [int(x) for x in re.findall(rf'\*\*{name} (\d+)', main)]
    if defs != list(range(1, len(defs) + 1)): errors.append(f'{name} numbering {defs}')
    for x in re.findall(rf'{name}s? (\d+)', main + supp):
        if int(x) > max(defs or [0]): errors.append(f'{name} {x} cited but not defined')
checks(main, 'main'); checks(supp, 'supp')
main = tags(main, 'main'); supp = tags(supp, 'supp')
open(f'{D}/manuscript_v4.md', 'w').write(main); open(f'{D}/supplement_v4.md', 'w').write(supp)
print('main tables', tn); print('main figures', fn); print('supp tables', stn); print('supp figures', sfn)
print('refs main', len(rm), 'supp-only', [k for k in rs if k not in rm])
print('ERRORS:' if errors else 'no errors'); [print(' -', e) for e in errors]
