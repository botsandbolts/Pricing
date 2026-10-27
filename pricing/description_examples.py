"""Real, price-redacted listing examples; never submit a valuation automatically."""
import json
from pathlib import Path


def load_examples():
    file = Path(__file__).resolve().parents[1] / 'data/description-examples.json'
    return json.loads(file.read_text())['examples'] if file.exists() else []


def render_examples(uses_text):
    import streamlit as st
    examples = load_examples()
    if not examples:
        return
    def apply():
        selected = st.session_state.get('description_example', 0)
        if not selected:
            return
        example = examples[selected - 1]
        st.session_state['seller_description'] = example['text']
        for key, value in example['fields'].items():
            if value is not None:
                st.session_state['vehicle_' + key] = value
        st.session_state.pop('last_explained_price', None)
        st.session_state.pop('pricing_visual_explanation', None)
    st.selectbox(
        f'Browse {len(examples):,} real car descriptions — type to search',
        range(len(examples) + 1), key='description_example', on_change=apply,
        format_func=lambda i: 'Write your own description' if i == 0 else examples[i - 1]['label'],
        disabled=not uses_text,
        help='Selecting an example loads its description and matching car fields. Edit either, then press Estimate asking price.')
    st.caption('Real historical listings, with advertised prices and contact details redacted. Selecting a car loads matching fields; nothing is priced until you click Estimate. Your keyboard remains the chief mechanic.')

    st.caption('Examples adapted from Tom, Used Cars Dataset (Zenodo 10457828), CC BY 4.0. Prices and common contact/VIN formats redacted; formatting normalized. [Dataset](https://zenodo.org/records/10457828) · [License](https://creativecommons.org/licenses/by/4.0/)')
