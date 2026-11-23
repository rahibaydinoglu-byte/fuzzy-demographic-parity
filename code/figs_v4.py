"""Re-render the manuscript figures into paper_v4/figures with the model named FairTSK (ZadehNet in the code)."""
import os, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.figure import Figure
OUT = os.path.join(os.path.dirname(__file__), '..', 'paper_v4', 'figures'); os.makedirs(OUT, exist_ok=True)
ren = lambda s: s.replace('ZadehNet-rule', 'FairTSK-rule').replace('ZadehNet-0', 'FairTSK-0').replace('ZadehNet0', 'FairTSK-0').replace('ZadehNet', 'FairTSK')
_save = Figure.savefig
def savefig(self, fname, *a, **k):
    for t in self.findobj(matplotlib.text.Text):
        if 'ZadehNet' in t.get_text(): t.set_text(ren(t.get_text()))
    return _save(self, os.path.join(OUT, os.path.basename(str(fname))), *a, **k)
Figure.savefig = savefig
import figs
for f in ['pareto', 'threshold', 'displacement', 'scaling', 'cd']: getattr(figs, f'{f}_fig')()
import runpy; runpy.run_path(os.path.join(os.path.dirname(__file__), 'fig_synth.py'))
print('figures written to', OUT)
