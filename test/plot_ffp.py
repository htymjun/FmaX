#!/usr/bin/env python3
"""Plot accuracy and throughput results from a fltflt test CSV.

Usage:  python3 plot_ffp.py <results.csv> <output.png>

CSV columns: type, label, val1, val2
  acc rows  — accuracy: val1=r4_err, val2=ff_err
  bench rows — throughput: val1=ms, val2=MOPS  (label: real(4)|real(8)|fltflt)
"""

import sys
import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


FLOOR = 1e-17   # display floor for zero-error results


def read_csv(path):
    acc   = []
    bench = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t     = row['type'].strip()
            label = row['label'].strip()
            v1    = float(row['val1'])
            v2    = float(row['val2'])
            if t == 'acc':
                acc.append((label, v1, v2))
            elif t == 'bench':
                bench[label] = (v1, v2)   # (ms, MOPS)
    return acc, bench


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <csv> <output.png>")
        sys.exit(1)

    csv_path = sys.argv[1]
    png_path = sys.argv[2]

    acc, bench = read_csv(csv_path)

    # derive operation name from CSV filename, e.g. "add_results.csv" -> "add"
    base    = os.path.splitext(os.path.basename(csv_path))[0]
    op_name = base.replace('_results', '').replace('ffp', 'MAD')

    # ── colour palette ────────────────────────────────────────────────
    C_R4  = '#D95F02'   # burnt orange  — real(4)
    C_FF  = '#1B7FCC'   # steel blue    — fltflt
    C_R8  = '#5A7A3A'   # muted olive   — real(8)
    BG    = '#F7F7F5'
    GRID  = '#DEDAD5'
    TEXT  = '#1A1A1A'

    if acc:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8),
                                       facecolor=BG,
                                       gridspec_kw={'hspace': 0.42})
        axes = (ax1, ax2)
    else:
        fig, ax2 = plt.subplots(1, 1, figsize=(7, 5), facecolor=BG)
        axes = (ax2,)

    for ax in axes:
        ax.set_facecolor(BG)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_color(GRID)

    # ══════════════════════════════════════════════════════════════════
    # Panel 1: accuracy (log-scale grouped bars) — only when acc data present
    # ══════════════════════════════════════════════════════════════════
    if acc:
        labels  = [r[0] for r in acc]
        r4_errs = [max(r[1], FLOOR) for r in acc]
        ff_errs = [max(r[2], FLOOR) for r in acc]
        x = np.arange(len(labels))
        w = 0.36

        ax1.bar(x - w/2, r4_errs, w, color=C_R4, label='real(4)', zorder=3)
        ax1.bar(x + w/2, ff_errs, w, color=C_FF, label='fltflt',  zorder=3)

        ax1.set_yscale('log')
        ax1.set_ylim(FLOOR * 0.3, 5.0)
        ax1.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=25, ha='right', fontsize=8.5)
        ax1.set_ylabel('Relative error', color=TEXT, fontsize=10)
        ax1.set_title(f'Accuracy  ({op_name}):  real(4) vs fltflt',
                      color=TEXT, fontsize=12, fontweight='bold', pad=8)
        ax1.yaxis.grid(True, which='both', color=GRID, linewidth=0.5, zorder=0)
        ax1.legend(framealpha=0, fontsize=9)

        for xi, row in enumerate(acc):
            if row[1] == 0.0:
                ax1.text(xi - w/2, FLOOR * 0.5, 'exact', ha='center',
                         va='top', fontsize=6.5, color=C_R4, rotation=90)
            if row[2] == 0.0:
                ax1.text(xi + w/2, FLOOR * 0.5, 'exact', ha='center',
                         va='top', fontsize=6.5, color=C_FF, rotation=90)

    # ══════════════════════════════════════════════════════════════════
    # Panel 2: throughput (MOPS bars — 2 or 3 depending on CSV content)
    # ══════════════════════════════════════════════════════════════════
    ms_r8, mops_r8 = bench.get('real(8)', (0.0, 0.0))
    ms_ff, mops_ff = bench.get('fltflt',  (0.0, 0.0))

    bx     = np.array([0.0, 1.0])
    mops   = [mops_r8, mops_ff]
    ms     = [ms_r8,   ms_ff]
    colors = [C_R8, C_FF]
    blabs  = ['real(8)', 'fltflt']

    bars = ax2.bar(bx, mops, 0.55, color=colors, zorder=3)

    for bar, m in zip(bars, ms):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 0.05,
                 f'{m:.2f} ms', ha='center', va='bottom',
                 fontsize=9, color='white', fontweight='bold')

    for bar, m in zip(bars, mops):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max(mops) * 0.01,
                 f'{m:,.0f}', ha='center', va='bottom',
                 fontsize=9, color=TEXT)

    if mops_r8 > 0 and mops_ff > 0:
        ratio     = mops_ff / mops_r8
        ratio_str = (f'fltflt {ratio:.2f}× faster' if ratio >= 1.0
                     else f'fltflt {1/ratio:.2f}× slower')
        ax2.annotate('', xy=(1.0, max(mops) * 1.03),
                     xytext=(0.0, max(mops) * 1.03),
                     arrowprops=dict(arrowstyle='<->', color=TEXT, lw=1.2))
        ax2.text(0.5, max(mops) * 1.08,
                 ratio_str, ha='center', va='bottom', fontsize=9, color=TEXT)

    ax2.set_xticks(bx)
    ax2.set_xticklabels(blabs, fontsize=10)
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(0, max(mops) * 1.25)
    ax2.set_ylabel('MOPS', color=TEXT, fontsize=10)
    ax2.set_title(
        rf'Throughput  ({op_name}):  chained operation  ($N=2^{{20}}$, 100-iter)',
        color=TEXT, fontsize=12, fontweight='bold', pad=8)
    ax2.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    fig.subplots_adjust(left=0.09, right=0.97, top=0.95, bottom=0.14, hspace=0.48)
    fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=BG)
    print(f"Saved: {png_path}")


if __name__ == '__main__':
    main()
