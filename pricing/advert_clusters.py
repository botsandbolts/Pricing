"""Unsupervised advert types: weighted Hellinger TF-IDF/category geometry.

Independent of every price model. Never interpret membership as seller identity.
"""
import hashlib
import html
import io
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from urllib.parse import quote, unquote

CATEGORIES = ('manufacturer', 'model', 'condition', 'cylinders', 'fuel',
              'title_status', 'transmission', 'drive', 'type', 'paint_color', 'state')
MISSING = {'', 'unknown', 'nan', 'none', 'null'}
NUMERIC = {'year': (1900, 2021), 'odometer': (0, 2000000)}


def numeric_values(records, key):
    values = pd.to_numeric(records.get(key, pd.Series(np.nan, index=records.index)), errors='coerce').to_numpy(dtype=float, copy=True)
    low, high = NUMERIC[key]
    values[(values < low) | (values > high)] = np.nan
    return values


def numeric_block(bundle, records):
    blocks = []
    for key in NUMERIC:
        values = numeric_values(records, key)
        knots = bundle['numeric_knots'][key]
        if len(knots) < 2:
            blocks.append(np.zeros((len(records), 2)))
            continue
        # Every observed number becomes a distribution [q, 1-q], retaining order.
        q = np.interp(values, knots, bundle['numeric_quantiles'][key])
        blocks.append(np.where(np.isfinite(values[:, None]), np.column_stack([q, 1-q]), 0))
    return sparse.csr_matrix(np.hstack(blocks), dtype=np.float32)


def clean_description(text):
    value = html.unescape(str(text or '')[:8000]).lower()
    value = re.sub(r'<[^>]*>|https?://\S+|www\.\S+|\S+@\S+', ' ', value)
    # Drop monetary values, phones, VINs and other numeric identifiers.
    value = re.sub(r'[$€£]\s*[\d,.]+|\b\w*\d\w*\b', ' ', value)
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z\s]', ' ', value)).strip()


def category_frame(records):
    return pd.DataFrame({key: records.get(key, pd.Series('', index=records.index))
                        .fillna('').astype(str).str.lower().str.strip()
                        for key in CATEGORIES}, index=records.index)


def categorical_tokens(records):
    """Field-qualified categorical values; preserve rare values and digits."""
    return [[f'cat:{key}={quote(value, safe="")}' for key, value in row.items()
             if value not in MISSING]
            for row in category_frame(records).to_dict(orient='records')]


def identity_tokens(tokens):
    return tokens


def token_documents(records, word_vectorizer):
    analyzer = word_vectorizer.build_analyzer()
    words = [analyzer(clean_description(text)) for text in records.text.fillna('')]
    return [w+c for w,c in zip(words, categorical_tokens(records))]


def encode(bundle, records):
    documents = token_documents(records, bundle['word_vectorizer'])
    tfidf = bundle['vectorizer'].transform(documents).astype(np.float32)
    numbers = numeric_block(bundle, records)
    # Categories share TF-IDF with language, with no separate dummy block.
    features = normalize(sparse.hstack([
        normalize(tfidf, norm='l1') * bundle['tfidf_weight'],
        normalize(numbers, norm='l1') * bundle['numeric_weight']
    ], format='csr'), norm='l1').astype(np.float32)
    usable = tfidf.getnnz(axis=1) >= 2
    return features, tfidf, usable


def distinctive_terms(tfidf, labels, terms, limit=6):
    """Contrast a cluster centroid with the corpus; prefer supported phrases."""
    global_mean = np.asarray(tfidf.mean(axis=0)).ravel()
    summaries = {}
    for label in sorted(set(labels) - {-1}):
        rows = tfidf[labels == label]
        centroid = np.asarray(rows.mean(axis=0)).ravel()
        support = np.asarray((rows > 0).mean(axis=0)).ravel()
        # Mean TF-IDF multiplied by smoothed log lift: distinctive AND prevalent.
        score = centroid * np.maximum(0, np.log((centroid + 1e-4) / (global_mean + 1e-4)))
        score[support < .08] = 0
        selected = []
        for index in np.argsort(score)[::-1]:
            term = str(terms[index])
            if score[index] <= 0:
                break
            words = set(term.split())
            if any(words <= set(other.split()) or set(other.split()) <= words for other in selected):
                continue
            selected.append(term)
            if len(selected) == limit:
                break
        if not selected:
            selected = [str(terms[i]) for i in np.argsort(centroid)[-limit:][::-1]]
        selected = [unquote(t[4:]).replace('=', ': ') if t.startswith('cat:') else t for t in selected]
        summaries[str(label)] = {'name': ' / '.join(t.title() for t in selected[:2]) + f' · C{label:03d}',
                                 'keywords': selected, 'size': int(rows.shape[0])}
    return summaries


class FrozenProjection:
    """Portable frozen UMAP geometry with exact Hellinger neighbour lookup.

    Save arrays, not Numba/NNDescent closures tied to a Python minor version.
    Use UMAP's precomputed-distance transform against the unchanged embedding.
    """
    def __init__(self, fitted):
        self.reference = fitted._raw_data.copy()
        self.reference.data = np.sqrt(self.reference.data)
        self.embedding = fitted.embedding_.copy()
        self.parameters = {key: getattr(fitted, key) for key in
                           ('_a', '_b', '_initial_alpha', '_n_neighbors', '_disconnection_distance')}
        self.seed = fitted.transform_seed

    def transform(self, features):
        import umap
        import warnings
        root = features.copy()
        root.data = np.sqrt(root.data)
        affinity = (root @ self.reference.T).toarray()
        distances = np.sqrt(np.maximum(0, 1-np.clip(affinity, 0, 1))).astype(np.float32)
        reducer = umap.UMAP(n_neighbors=self.parameters['_n_neighbors'], n_components=self.embedding.shape[1],
                            metric='precomputed', random_state=self.seed, transform_seed=self.seed,
                            min_dist=0, n_jobs=1, tqdm_kwds={})
        reducer.embedding_ = self.embedding
        reducer._raw_data = sparse.csr_matrix((self.embedding.shape[0], 0), dtype=np.float32)
        reducer._input_hash = None
        for key, value in self.parameters.items():
            setattr(reducer, key, value)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Transforming new data with precomputed metric.*')
            return reducer.transform(distances)


def fit(records, *, seed=42, n_neighbors=10, n_components=5, min_dist=0.,
        min_cluster_size=15, min_samples=1):
    import umap
    import hdbscan
    if len(records) < 100:
        raise ValueError('At least 100 training descriptions are required')
    records = records.copy()
    records['text'] = records['text'].fillna('').map(clean_description)
    word_vectorizer = TfidfVectorizer(min_df=5, max_df=.95, max_features=16000,
                                ngram_range=(1, 2), stop_words='english',
                                sublinear_tf=True, dtype=np.float32)
    word_vectorizer.fit(records.text)
    vocabulary = set(word_vectorizer.get_feature_names_out())
    vocabulary.update(token for row in categorical_tokens(records) for token in row)
    # Fixed vocabulary retains EVERY observed categorical level, including singletons.
    vectorizer = TfidfVectorizer(analyzer=identity_tokens, vocabulary=sorted(vocabulary), lowercase=False,
                                min_df=1, sublinear_tf=True, dtype=np.float32)
    vectorizer.fit(token_documents(records, word_vectorizer))
    numeric_knots, numeric_quantiles = {}, {}
    for key in NUMERIC:
        values = numeric_values(records, key)
        values = values[np.isfinite(values)]
        probabilities = np.linspace(0, 1, 257)
        quantiles = np.quantile(values, probabilities) if len(values) else np.array([])
        knots = np.unique(quantiles)
        numeric_knots[key] = knots
        numeric_quantiles[key] = np.array([probabilities[np.flatnonzero(quantiles == k)[-1]] for k in knots])
    bundle = dict(word_vectorizer=word_vectorizer, vectorizer=vectorizer, tfidf_weight=.85,
                  numeric_weight=.15,
                  numeric_knots=numeric_knots, numeric_quantiles=numeric_quantiles)
    features, tfidf, usable = encode(bundle, records)
    features = features[usable]; tfidf = tfidf[usable]
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist,
                        metric='hellinger', random_state=seed, transform_seed=seed,
                        n_jobs=1, low_memory=True)
    embedding = reducer.fit_transform(features)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                               metric='euclidean', prediction_data=True, core_dist_n_jobs=1)
    labels = clusterer.fit_predict(embedding)
    if len(set(labels) - {-1}) < 2:
        raise ValueError('Fewer than two clusters found; do not deploy a collapsed partition')
    names = distinctive_terms(tfidf, labels, vectorizer.get_feature_names_out())
    for label, summary in names.items():
        members = embedding[labels == int(label)]
        centroid = members.mean(axis=0)
        # A real member nearest the centroid, rather than an invented sentence.
        indices = np.flatnonzero(labels == int(label))
        representative = int(indices[np.argmin(np.linalg.norm(members - centroid, axis=1))])
        text = records.loc[usable, 'text'].iloc[representative]
        summary['representative_sha256'] = hashlib.sha256(text.encode()).hexdigest()
        summary['mean_membership_strength'] = float(clusterer.probabilities_[labels == int(label)].mean())
    bundle.update(reducer=FrozenProjection(reducer), clusterer=clusterer, clusters=names, schema_version=2,
                  min_strength=0.)
    report = dict(training_rows=int(len(labels)), clusters=names, cluster_count=len(names),
                  training_noise_fraction=float(np.mean(labels == -1)),
                  categorical_fields=list(CATEGORIES), category_min_count=1,
                  categorical_token_count=sum(t.startswith('cat:') for t in vectorizer.vocabulary_),
                  singleton_categorical_tokens=sum(count == 1 for count in Counter(t for row in categorical_tokens(records) for t in row).values()),
                  tfidf_weight=.85, numeric_weight=bundle['numeric_weight'],
                  numeric_fields=list(NUMERIC), numeric_transform='training quantile rank encoded as [q, 1-q]',
                  distance='weighted block-normalized Hellinger',
                  umap=dict(n_neighbors=n_neighbors, n_components=n_components, min_dist=min_dist, metric='hellinger', seed=seed),
                  hdbscan=dict(min_cluster_size=min_cluster_size, min_samples=min_samples),
                  naming='centroid TF-IDF × smoothed positive log lift; minimum 8% cluster term support',
                  minimum_assignment_strength=bundle['min_strength'])
    return bundle, report


def classify(bundle, text, fields):
    import hdbscan
    empty = dict(cluster_id=-1, name='Unclassified advert', keywords=[], strength=0.)
    record = pd.DataFrame([{**fields, 'text': text}])
    features, _, usable = encode(bundle, record)
    if not usable[0]:
        return {**empty, 'reason': 'Add recognizable description words or more vehicle categories.'}
    embedding = bundle['reducer'].transform(features)
    labels, strength = hdbscan.approximate_predict(bundle['clusterer'], embedding)
    label, value = int(labels[0]), float(strength[0])
    if label < 0 or value < bundle['min_strength']:
        return {**empty, 'strength': value, 'reason': 'HDBSCAN found no matching density group in the historical adverts.'}
    group = bundle['clusters'][str(label)]
    return dict(cluster_id=label, name=group['name'], keywords=group['keywords'],
                strength=value, training_members=group['size'], reason=None)


def load_bundle(folder):
    """Only load this repository's trusted, checksum-verified pickle artifact."""
    import joblib
    from importlib.metadata import version
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    for package, expected in manifest['versions'].items():
        actual = version(package)
        if actual != expected:
            raise RuntimeError(f'Advert groups require {package} {expected}; installed {actual}')
    from pricing.suite_storage import read_model_file
    payload = read_model_file(folder/'model.joblib')
    if hashlib.sha256(payload).hexdigest() != manifest['sha256']:
        raise ValueError('Advert cluster artifact checksum mismatch')
    return joblib.load(io.BytesIO(payload))
