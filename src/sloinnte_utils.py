import xml.etree.ElementTree as ET
import pandas as pd
import networkx as nx
from sklearn.model_selection import GroupShuffleSplit
import re
import unicodedata
import numpy as np
from itertools import combinations
from rapidfuzz.distance import Levenshtein

IRISH_PREFIXES = {"o", "ua", "ui", "ni", "mac", "mag", "mhic", "nic", "de", "bean", "an"}
FEATURES = ["edit_dist", "len_diff", "len_ratio", "shared_chars", "same_first_letter", "cross_lang"]


def load_forms(path):
    """Parse the Gaois XML into one row per form, removing empty forms."""
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for cluster_id, cluster in enumerate(root.findall("cluster")):
        title = cluster.findtext("title")
        for section in cluster:
            # section tags look like "surnames-irish-main"
            if not section.tag.startswith("surnames-"):
                continue
            _, language, status = section.tag.split("-")
            for surname in section:
                for form in surname.findall("form"):
                    rows.append({
                        "cluster_id": cluster_id,
                        "title": title,
                        "form": (form.text or "").strip(),
                        "language": language,   # irish / english
                        "status": status,       # main / historical / alternative
                    })

    df = pd.DataFrame(rows)
    df = df[df["form"] != ""].reset_index(drop=True)
    return df

def add_group_ids(df):
    """Add a group_id linking clusters that share any form (connected components)."""
    df = df.copy()
    G = nx.Graph()
    for cid, form in df[["cluster_id", "form"]].itertuples(index=False):
        G.add_edge(("cluster", cid), ("form", form))

    cluster_to_group = {}
    for gid, component in enumerate(nx.connected_components(G)):
        for kind, value in component:
            if kind == "cluster":
                cluster_to_group[value] = gid

    df["group_id"] = df["cluster_id"].map(cluster_to_group)
    return df

def split_train_test(df, test_size=0.2, seed=42):
    """Split forms into train and test sets by group_id, so no group (or form) appears in both."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["group_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, test_df

def make_pairs(split_df, neg_ratio=1, seed=42):
    """Build positive and negative pairs using ONLY the forms in split_df."""
    rng = np.random.default_rng(seed)
    form_clusters = split_df.groupby("form")["cluster_id"].apply(set).to_dict()

    # Positives: two forms in the same cluster, at least one Irish
    pos = {}
    for cid, g in split_df.groupby("cluster_id"):
        items = list(g[["form", "language"]].itertuples(index=False, name=None))
        for (f1, l1), (f2, l2) in combinations(items, 2):
            if f1 == f2 or "irish" not in (l1, l2):
                continue
            pos[frozenset([f1, f2])] = "ga-ga" if l1 == l2 else "ga-en"

    # Negatives: forms from different clusters, same pair type as each positive
    irish = split_df.loc[split_df["language"] == "irish", "form"].unique()
    english = split_df.loc[split_df["language"] == "english", "form"].unique()
    neg = {}
    for ptype in pos.values():
        made = 0
        while made < neg_ratio:
            a = rng.choice(irish)
            b = rng.choice(irish if ptype == "ga-ga" else english)
            key = frozenset([a, b])
            if a == b or key in pos or key in neg or form_clusters[a] & form_clusters[b]:
                continue
            neg[key] = ptype
            made += 1

    rows = [(*sorted(k), t, 1) for k, t in pos.items()] + \
           [(*sorted(k), t, 0) for k, t in neg.items()]
    return pd.DataFrame(rows, columns=["form_a", "form_b", "pair_type", "is_match"])

def normalise(s):
    """Lowercase, strip whitespace and remove fadas/diacritics from a surname form."""
    s = unicodedata.normalize("NFD", s.lower().strip())
    return "".join(c for c in s if unicodedata.category(c) != "Mn")  # removes fadas

def strip_prefix(s):
    """Remove leading Irish prefix words (Ó, Mac, Ní, etc.) and English O'/Mc from a normalised form."""
    parts = s.split()
    while len(parts) > 1 and parts[0] in IRISH_PREFIXES:
        parts = parts[1:]
    s = " ".join(parts)
    return re.sub(r"^(o'|mc)", "", s)

def add_features(pairs):
    """Return a copy of the pairs DataFrame with string-similarity features added."""
    pairs = pairs.copy()
    a = pairs["form_a"].map(normalise)
    b = pairs["form_b"].map(normalise)
    la, lb = a.str.len(), b.str.len()
    pairs["edit_dist"] = [Levenshtein.normalized_distance(x, y) for x, y in zip(a, b)]
    pairs["len_diff"] = (la - lb).abs()
    pairs["len_ratio"] = np.minimum(la, lb) / np.maximum(la, lb)
    pairs["shared_chars"] = [len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(a, b)]
    pairs["same_first_letter"] = [int(strip_prefix(x)[:1] == strip_prefix(y)[:1]) for x, y in zip(a, b)]
    pairs["cross_lang"] = (pairs["pair_type"] == "ga-en").astype(int)
    return pairs

