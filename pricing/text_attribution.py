"""Bounded, direct grouped permutation Shapley explanations in served dollars.

No surrogate, training labels, global cache, or third-party explanation runtime.
The callable must rerun the entire text-to-dollar pipeline with car facts fixed.
"""
from itertools import permutations
import hashlib
import html
import json
import re

import numpy as np

VERSION = 'phrase-shap-v1'
DETAILS = {'Quick look': (8, 2), 'Closer look': (12, 4)}


def phrase_groups(text, limit=8):
    """Prefer sentence/clause boundaries; retain every character in order."""
    if not 1 <= limit <= 12:
        raise ValueError('Use between 1 and 12 phrase groups')
    words = re.findall(r'\S+\s*', text)
    if not words:
        return []
    clauses, current = [], []
    for word in words:
        current.append(word)
        if word.rstrip().endswith(('.', '!', '?', ';')):
            clauses.append(current)
            current = []
    if current:
        clauses.append(current)
    if len(clauses) > 1:
        # Merge the shortest neighbours if there are too many sentences.
        while len(clauses) > limit:
            i = min(range(len(clauses)-1), key=lambda i: len(clauses[i])+len(clauses[i+1]))
            clauses[i:i+2] = [clauses[i]+clauses[i+1]]
        # Keep short sentences intact; bound the size of lengthy paragraphs.
        while len(clauses) < limit and max(map(len, clauses)) > 12:
            i = max(range(len(clauses)), key=lambda i: len(clauses[i]))
            cut = (len(clauses[i])+1)//2
            clauses[i:i+1] = [clauses[i][:cut], clauses[i][cut:]]
        return [''.join(clause) for clause in clauses]
    groups = np.array_split(np.arange(len(words)), min(limit, len(words)))
    return [''.join(words[int(i)] for i in group) for group in groups]


def signature(manifest, fields, text, detail, instance, price):
    body = [VERSION, manifest, fields, text, detail, instance, price]
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def explain_text(predict, text, *, limit=8, pairs=2, seed=42, progress=None):
    """Average forward/reverse permutation paths; exact for <=4 groups.

    Missing groups are deleted and separated by spaces. All coalitions keep
    original ordering. Empty-description baseline is an actual model prediction.
    SE describes sampled-order variability, never valuation uncertainty.
    """
    if not 2 <= pairs <= 8:
        raise ValueError('Use between 2 and 8 permutation pairs')
    groups = phrase_groups(text, limit)
    n = len(groups)
    if not n:
        raise ValueError('Enter a description before explaining its phrases')
    full = (1 << n) - 1
    exact = n <= 4
    if exact:
        orders = list(permutations(range(n)))
    else:
        rng = np.random.default_rng(seed)
        orders = []
        for _ in range(pairs):
            order = rng.permutation(n).tolist()
            orders.extend([order, order[::-1]])
    masks = {0, full}
    for order in orders:
        mask = 0
        for i in order:
            mask |= 1 << i
            masks.add(mask)
    outputs, predictions = {}, {}
    for number, mask in enumerate(sorted(masks)):
        candidate = text if mask == full else ' '.join(
            group.strip() for i, group in enumerate(groups) if mask & (1 << i))
        if candidate not in predictions:
            value = float(predict(candidate))
            if not np.isfinite(value) or value < 0:
                raise ValueError('The pricing model returned an invalid dollar estimate')
            predictions[candidate] = value
        outputs[mask] = predictions[candidate]
        if progress:
            progress(number + 1, len(masks))
    paths = []
    for order in orders:
        mask, values = 0, np.zeros(n)
        for i in order:
            next_mask = mask | (1 << i)
            values[i] = outputs[next_mask] - outputs[mask]
            mask = next_mask
        paths.append(values)
    paths = np.asarray(paths)
    values = paths.mean(axis=0)
    pair_values = paths.reshape(pairs, 2, n).mean(axis=1) if not exact else None
    se = np.zeros(n) if exact else pair_values.std(axis=0, ddof=1) / np.sqrt(pairs)
    error = abs(outputs[0] + float(values.sum()) - outputs[full])
    if error > max(1e-5, abs(outputs[full]) * 1e-10):
        raise ValueError('Text explanation failed its dollar reconstruction check')
    return {'version': VERSION, 'groups': groups, 'values': values.tolist(),
            'se': se.tolist(), 'base': outputs[0], 'prediction': outputs[full],
            'difference': outputs[full] - outputs[0], 'error': error,
            'orders': len(orders), 'evaluations': len(predictions), 'exact': exact,
            'seed': seed, 'baseline': 'Same vehicle facts; empty description',
            'masking': 'Delete missing phrase groups; preserve original order',
            'scope': 'Historical asking-price model sensitivity; not causal market value'}


def highlight_html(result):
    """Escaped, accessible phrase highlights with signed dollar tooltips."""
    maximum = max(max(abs(v) for v in result['values']), 1.)
    spans = []
    for i, (group, value) in enumerate(zip(result['groups'], result['values'])):
        colour = '166,54,33' if value > .005 else '25,82,169' if value < -.005 else '100,100,100'
        alpha = .09 + .20 * abs(value) / maximum
        title = html.escape(f'Phrase {i+1}: {value:+,.2f} USD allocated contribution', quote=True)
        spans.append(f'<span title="{title}" style="background:rgba({colour},{alpha:.3f});'
                     f'border-bottom:3px solid rgb({colour});border-radius:5px;padding:3px 2px;'
                     f'box-decoration-break:clone">{html.escape(group)}</span>')
    return ('<div style="font-size:1.06rem;line-height:2.2;overflow-wrap:anywhere;'
            'white-space:pre-wrap">' + ''.join(spans) + '</div>')


def contribution_chart(result, show_all=False):
    import plotly.graph_objects as go
    ranked = sorted(range(len(result['groups'])), key=lambda i: abs(result['values'][i]), reverse=True)
    visible = ranked if show_all else ranked[:6]
    rows = [(result['groups'][i].strip(), result['values'][i]) for i in visible]
    if len(visible) < len(ranked):
        rows.append(('Other phrases · combined', sum(result['values'][i] for i in ranked[6:])))
    labels = [f'{i+1}. {html.escape(label[:65])}{"…" if len(label)>65 else ""}' for i, (label, _) in enumerate(rows)]
    values = [v for _, v in rows]
    fig = go.Figure(go.Bar(y=labels, x=values, orientation='h',
        marker_color=['#a63621' if v > 0 else '#1952a9' for v in values],
        text=[f'{v:+,.0f} USD' for v in values], textposition='auto',
        customdata=[html.escape(label) for label, _ in rows],
        hovertemplate='%{customdata}<br>%{x:+,.2f} USD<extra></extra>'))
    fig.update_layout(height=max(310, 42*len(rows)+90), margin=dict(l=0, r=15, t=10, b=30),
        xaxis_title='Allocated effect on asking estimate (USD)',
        yaxis=dict(autorange='reversed'), showlegend=False)
    fig.add_vline(x=0, line_width=1, line_color='#777777')
    return fig
