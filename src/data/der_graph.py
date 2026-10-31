"""
Shared DER (Domain Entity Recognition) -> graph construction.

This is the SINGLE canonical version of this code, extracted verbatim from
Copy_of_Graph_models.ipynb (cells 9-15), after confirming cell-by-cell that it is
byte-identical to the RGAT notebook's version (notebookB_RGAT_base_reg_raw_collapsed.ipynb).

The baselines notebook previously used an older, incompatible version of
build_sentence_graph/attach_surfaces (nodes keyed by span text only, isolated DER
nodes silently dropped if they took part in no relation). That has already been
fixed in the baselines notebook to match this one -- see git history. Every
notebook (baselines, GAT/R-GCN, RGAT, qualitative analysis) should import from
this module going forward instead of redefining these functions inline, so the
DER graphs used by every model are guaranteed identical.

Usage:
    from src.data.der_graph import RuleBasedRelations, build_dataset

    extractor = RuleBasedRelations()
    DATA = {name: build_dataset(d["pos"], d["neg"], extractor)
            for name, d in DATASETS.items()}
"""
import itertools
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    _nlp = None  # falls back to simple plural stripping


# ═══════════════════════ Identity normalisation ═══════════════════════
# ("Hindus" -> "hindu", "OBC family" -> "obc", ...)

_IDENTITY_MODIFIERS = frozenset({
    "family", "families", "group", "community", "commnunity",
    "people", "person", "man", "woman", "child", "children"
})
_PLURAL_OVERRIDES = {
    "hindus": "hindu", "muslims": "muslim", "vaishyas": "vaishya", "brahmins": "brahmin",
    "dalits": "dalit", "obcs": "obc", "christians": "christian", "sikhs": "sikh",
    "jains": "jain", "buddhists": "buddhist",
}
_COMPOUND_PATTERNS = [
    (r"\b(obc|hindu|muslim|brahmin|dalit|vaishya|christian|sikh|jain)\s+(family|group|community|people)\b", r"\1"),
    (r"\b(hindu|muslim|brahmin|dalit|vaishya)\s+(man|woman|person|child)\b", r"\1"),
]


def _strip_plural(text):
    return text[:-1] if text.endswith("s") and not text.endswith(("ss", "us", "is")) else text


def _normalize_identity(span: str) -> str:
    text = span.lower().strip()
    for pattern, repl in _COMPOUND_PATTERNS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    words = text.split()
    if len(words) > 1 and words[-1] in _IDENTITY_MODIFIERS:
        text = " ".join(words[:-1])
    if text in _PLURAL_OVERRIDES:
        text = _PLURAL_OVERRIDES[text]
    if _nlp is not None:
        try:
            text = " ".join(t.lemma_ for t in _nlp(text))
        except Exception:
            text = _strip_plural(text)
    else:
        text = _strip_plural(text)
    return text.strip()


# ═══════════════════════ Identity -> category mapping ═══════════════════════

_IDENTITY_TO_CATEGORY = {
    # Religion
    "hindu": "Religion", "muslim": "Religion", "christian": "Religion", "sikh": "Religion",
    "jain": "Religion", "buddhist": "Religion", "islam": "Religion", "christianity": "Religion",
    "hinduism": "Religion", "sikhism": "Religion", "parsi": "Religion", "parsis": "Religion",
    "buddhism": "Religion", "jainism": "Religion",
    # Caste
    "brahmin": "Caste", "vaishya": "Caste", "dalit": "Caste", "obc": "Caste", "sc": "Caste",
    "st": "Caste", "kshatriya": "Caste", "shudra": "Caste", "sc / st": "Caste", "kaystha": "Caste",
    "bania": "Caste", "kurmi": "Caste", "jat": "Caste", "rajput": "Caste", "adivasi": "Caste",
    "kayast": "Caste", "general": "Caste", "forward class": "Caste", "artisan": "Caste",
    "merchant": "Caste", "trader": "Caste", "laborer": "Caste", "peasant": "Caste",
    # Gender
    "women": "Gender", "man": "Gender", "woman": "Gender", "female": "Gender", "male": "Gender",
    "boy": "Gender", "girl": "Gender", "girls": "Gender", "boys": "Gender", "men": "Gender",
    "transgender male": "Gender", "lesbian": "Gender",
}


# ══════════════════════════ 1. Relation extraction ══════════════════════════

class RuleBasedRelations:
    """Converts a DER annotation dict into a flat list of canonical edges."""

    # ── Node-type labels ──────────────────────────────────────────────────
    IDENTITY_NODE   = "Identity Dimensions"
    PROFESSION_NODE = "Profession"
    TRIGGER_NODE    = "Contextual Trigger"
    FRAME_NODE      = "Situational Frame"

    # ── Canonical relation-type vocabulary ────────────────────────────────
    REL_IDENTITY_INTERSECTION     = "identity_intersection"
    REL_IDENTITY_ROLE_ASSOCIATION = "identity_role_association"
    REL_CHARACTERIZED_AS          = "characterized_as"
    REL_IDENTITY_FRAME_ASSOCIATION = "identity_frame_association"

    # ── Copula / auxiliary / modal verbs ──────────────────────────────────
    PREDICATE_LEAD_WORDS: frozenset = frozenset({
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did",
        "has", "have", "had",
        "can", "could", "will", "would", "shall", "should",
        "may", "might", "must",
        "seem", "seems", "seemed",
        "appear", "appears", "appeared",
        "become", "becomes", "became",
        "get", "gets", "got",
        "remain", "remains", "remained",
    })

    @staticmethod
    def _clean(span: Any) -> str:
        return span.strip() if isinstance(span, str) else ""

    @staticmethod
    def _normalize(span: str) -> str:
        return _normalize_identity(span)

    # ── STEP 1 — Node extraction ────────────────────────────────────────────
    def extract_nodes(self, annotation: Dict[str, Any]) -> Dict[str, List[Any]]:
        """Bucket the flat ``annotations`` list into typed groups."""
        identities: List[Dict[str, Any]] = []
        professions: List[str] = []
        triggers: List[str] = []
        frames: List[str] = []

        for ann in annotation.get("annotations", []):
            raw = self._clean(ann.get("span", ""))
            if not raw:
                continue
            ntype = ann.get("node")
            span = self._normalize(raw) if ntype == self.IDENTITY_NODE else raw.lower().strip()

            if ntype == self.IDENTITY_NODE:
                identities.append({"span": span, "label": ann.get("label"),
                                   "category": _IDENTITY_TO_CATEGORY.get(span, "Other")})
            elif ntype == self.PROFESSION_NODE:
                professions.append(span)
            elif ntype == self.TRIGGER_NODE:
                triggers.append(span)
            elif ntype == self.FRAME_NODE:
                frames.append(span)

        return {"identities": identities, "professions": professions,
               "triggers": triggers, "frames": frames}

    # ── STEP 2 — Frame classification ───────────────────────────────────────
    def _is_copula_frame(self, frame: str) -> bool:
        """True if frame is a copula/auxiliary verb (meaningless on its own)."""
        tokens = frame.strip().lower()
        return bool(tokens) and tokens in self.PREDICATE_LEAD_WORDS

    # ── STEP 3 — Relation extraction ────────────────────────────────────────
    def extract_relations(self, nodes):
        identities = nodes["identities"]
        professions = nodes["professions"]
        triggers = nodes["triggers"]
        frames = nodes["frames"]
        relations = []

        # Identity <-> Identity
        if len(identities) > 1:
            relations.extend(self._build_intersections(identities))

        # Identity <-> Profession
        if professions:
            relations.extend(self._build_role_associations(identities, professions))

        # Frame-driven patterns
        if len(frames) == 1 and len(triggers) > 1:
            frame_pairs = [(t, frames[0]) for t in triggers]
        else:
            frame_pairs = list(zip(triggers, frames))

        for trigger, frame in frame_pairs:
            if self._is_copula_frame(frame):
                relations.extend(self._build_predicate_relations(identities, professions, trigger, frame))
            else:
                relations.extend(self._build_semantic_relations(identities, professions, trigger, frame))

        return self._dedupe(relations)

    # ── Relation builders ────────────────────────────────────────────────────
    def _build_intersections(self, identities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        return [self._edge(a["span"], b["span"], self.REL_IDENTITY_INTERSECTION)
               for a, b in itertools.combinations(identities, 2)]

    def _build_role_associations(self, identities, professions) -> List[Dict[str, str]]:
        return [self._edge(i["span"], p, self.REL_IDENTITY_ROLE_ASSOCIATION)
               for i in identities for p in professions]

    def _build_predicate_relations(self, identities, professions, trigger, frame):
        relations = []
        for identity in identities:
            relations.append(self._edge(identity["span"], trigger, frame))
        for profession in professions:
            relations.append(self._edge(profession, trigger, frame))
        return relations

    def _build_semantic_relations(self, identities, professions, trigger, frame):
        relations = []
        for identity in identities:  # Identity -> Frame
            relations.append(self._edge(identity["span"], frame, self.REL_IDENTITY_FRAME_ASSOCIATION))
        for profession in professions:  # Profession -> Frame
            relations.append(self._edge(profession, frame, self.REL_IDENTITY_ROLE_ASSOCIATION))
        relations.append(self._edge(frame, trigger, self.REL_CHARACTERIZED_AS))  # Frame -> Trigger
        return relations

    # ── Utilities ────────────────────────────────────────────────────────────
    @staticmethod
    def _edge(source: str, target: str, relation: str) -> Dict[str, str]:
        return {"source": source, "target": target, "relation": relation}

    @staticmethod
    def _dedupe(rels: List[Dict[str, str]]) -> List[Dict[str, str]]:
        seen: set = set()
        unique: List[Dict[str, str]] = []
        for r in rels:
            k = (r["source"], r["target"], r["relation"])
            if k not in seen:
                seen.add(k)
                unique.append(r)
        return unique

    # ── Public entry points ───────────────────────────────────────────────
    def process_annotation(self, annotation: Dict[str, Any]) -> List[Dict[str, str]]:
        nodes = self.extract_nodes(annotation)
        return self.extract_relations(nodes)

    def process_batch(self, annotations: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for ann in annotations:
            out.extend(self.process_annotation(ann))
        return out


# ══════════════ 2. Per-sentence graph: typed nodes, safe empty-annotation handling ══════════════
# Each graph node is keyed by (normalised span, node type), so identical text assigned to
# different DER node types is preserved as distinct typed nodes. Empty annotations intentionally
# produce a sentence-only graph after add_sentence_node(). THIS IS THE FIX: the earlier baselines
# version keyed nodes by span text alone and dropped isolated DER spans (ones in no relation).

def _normalise_annotated_span(raw_span: str, node_type: str) -> str:
    raw = raw_span.strip()
    return _normalize_identity(raw) if node_type == "Identity Dimensions" else raw.lower().strip()


def _typed_node_id(span: str, node_type: str):
    return (span, node_type)


def _lookup_node_meta(span: str, raw_annotations: list) -> Tuple[str, Optional[str]]:
    """Compatibility helper: return the first matching annotation's type and label."""
    for ann in raw_annotations:
        raw = ann.get("span", "")
        if not isinstance(raw, str):
            continue
        node_type = ann.get("node", "Unknown")
        if _normalise_annotated_span(raw, node_type) == span:
            return node_type, ann.get("label")
    return "Unknown", None


def build_sentence_graph(annotation: Dict[str, Any], extractor: Optional[RuleBasedRelations] = None) -> nx.DiGraph:
    if extractor is None:
        extractor = RuleBasedRelations()

    G = nx.DiGraph()
    raw_anns = annotation.get("annotations", []) or []

    # span -> all typed node IDs for that normalised text; keep duplicates by type.
    candidates = defaultdict(list)

    # 1. Add every valid annotation, including isolated nodes.
    for ann in raw_anns:
        raw_span = ann.get("span", "")
        if not isinstance(raw_span, str) or not raw_span.strip():
            continue
        node_type = ann.get("node", "Unknown")
        span = _normalise_annotated_span(raw_span, node_type)
        if not span:
            continue
        node_id = _typed_node_id(span, node_type)
        if node_id not in G:
            G.add_node(node_id, node_type=node_type, ontology_label=ann.get("label"),
                      frequency=1, surface=raw_span.strip())
            candidates[span].append(node_id)
        else:
            G.nodes[node_id]["frequency"] += 1

    # Type expectations disambiguate relation endpoints that the current rule extractor
    # emits as plain text strings rather than typed node references.
    ID, PROF, TRIG, FRAME = (RuleBasedRelations.IDENTITY_NODE, RuleBasedRelations.PROFESSION_NODE,
                             RuleBasedRelations.TRIGGER_NODE, RuleBasedRelations.FRAME_NODE)
    expected_types = {
        RuleBasedRelations.REL_IDENTITY_INTERSECTION: ({ID}, {ID}),
        RuleBasedRelations.REL_CHARACTERIZED_AS: ({FRAME}, {TRIG}),
        RuleBasedRelations.REL_IDENTITY_FRAME_ASSOCIATION: ({ID}, {FRAME}),
        RuleBasedRelations.REL_IDENTITY_ROLE_ASSOCIATION: ({ID, PROF}, {PROF, FRAME}),
    }

    def resolve_endpoint(text, preferred_types=None):
        text = str(text).strip()
        matches = candidates.get(text, [])
        if preferred_types:
            preferred = [n for n in matches if G.nodes[n].get("node_type") in preferred_types]
            if preferred:
                matches = preferred
        if matches:
            return matches[0]
        node_id = _typed_node_id(text, "Unknown")
        if node_id not in G:
            G.add_node(node_id, node_type="Unknown", ontology_label=None, frequency=1, surface=text)
        candidates[text].append(node_id)
        return node_id

    # 2. Add extracted relations, mapped onto type-aware node IDs.
    for rel in extractor.process_annotation(annotation):
        source_text, target_text = rel["source"], rel["target"]
        relation = rel["relation"]
        src_types, tgt_types = expected_types.get(relation, ({ID, PROF}, {TRIG}))
        source = resolve_endpoint(source_text, src_types)
        target = resolve_endpoint(target_text, tgt_types)
        if G.has_edge(source, target):
            G[source][target]["weight"] += 1
        else:
            G.add_edge(source, target, relation=relation, weight=1)

    G.graph["sentence"] = annotation.get("sentence", "")
    return G


# ══════════════════════════ 3. Sentence node ══════════════════════════

SENTENCE_NODE_TYPE = "Sentence"
SENTENCE_RELATION = "contains"


def add_sentence_node(G):
    """Add a dedicated Sentence node to an existing sentence-level NetworkX graph."""
    G = G.copy()
    sentence = G.graph.get("sentence", "")
    sentence_node = "__SENTENCE__"
    G.add_node(sentence_node, node_type=SENTENCE_NODE_TYPE, ontology_label=None, frequency=1, text=sentence)
    for node in list(G.nodes()):
        if node == sentence_node:
            continue
        G.add_edge(sentence_node, node, relation=SENTENCE_RELATION, weight=1)
    return G


# ══════════════════════════ 4. Surface-text attachment ══════════════════════════
# Needed for contextual BERT alignment (BERT-CTX locates spans in the raw sentence).

IDENTITY_NODE_NAME = "Identity Dimensions"


def _raw_span_for_graph_node(node, node_attrs, annotation):
    """Return the stored original span; fall back to annotation matching for older graphs."""
    if node_attrs.get("surface"):
        return str(node_attrs["surface"])
    ntype = node_attrs.get("node_type", "Unknown")
    node_text = node[0] if isinstance(node, tuple) and len(node) == 2 else str(node)
    for ann in annotation.get("annotations", []):
        if ann.get("node", "Unknown") != ntype:
            continue
        raw = str(ann.get("span", "")).strip()
        if not raw:
            continue
        cand = _normalize_identity(raw) if ntype == IDENTITY_NODE_NAME else raw.lower().strip()
        if cand == str(node_text).lower().strip():
            return raw
    return str(node_text)


def attach_surfaces(G, annotation):
    for n, a in G.nodes(data=True):
        if a.get("node_type") == SENTENCE_NODE_TYPE:
            continue
        a["surface"] = _raw_span_for_graph_node(n, a, annotation)
    return G


# ══════════════════════════ 5. Relation families + NetworkX-level edits ══════════════════════════
# Used by ablations / occlusion (counterfactual and deletion experiments have been dropped from
# the pipeline -- see project decision log).

IDENTITY_NODE = RuleBasedRelations.IDENTITY_NODE
CANONICAL_RELATIONS = {
    RuleBasedRelations.REL_IDENTITY_INTERSECTION,
    RuleBasedRelations.REL_IDENTITY_ROLE_ASSOCIATION,
    RuleBasedRelations.REL_CHARACTERIZED_AS,
    RuleBasedRelations.REL_IDENTITY_FRAME_ASSOCIATION,
    SENTENCE_RELATION,
}


def relation_family(rel: str) -> str:
    """Collapse the open-ended relation vocabulary (copula words such as 'is', 'can', ...)
    into 'predicate'; canonical DER relations and 'contains' are kept. Strips 'rev_'."""
    r = rel[4:] if rel.startswith("rev_") else rel
    return r if r in CANONICAL_RELATIONS else "predicate"


def nx_remove_der_edges(G, families=None):
    """Remove DER edges (everything except Sentence->node 'contains').
    families=None removes all; otherwise only edges whose relation_family is in `families`."""
    G2 = G.copy()
    drop = [(u, v) for u, v, a in G2.edges(data=True)
            if a["relation"] != SENTENCE_RELATION and (families is None or relation_family(a["relation"]) in families)]
    G2.remove_edges_from(drop)
    return G2


def nx_remove_node_type(G, node_type):
    G2 = G.copy()
    G2.remove_nodes_from([n for n, a in G2.nodes(data=True) if a.get("node_type") == node_type])
    return G2


# ══════════════════════════ 6. Dataset assembly ══════════════════════════
# Builds the graphs/labels/groups arrays that EVERY notebook's make_splits() consumes. Keeping
# this shared too (not just the graph functions) guarantees every notebook gets the same labels
# and the same CV groups for the same annotation files, not just the same per-sentence graphs.

def build_dataset(annos_pos, annos_neg, extractor: Optional[RuleBasedRelations] = None):
    if extractor is None:
        extractor = RuleBasedRelations()
    annos = list(annos_pos) + list(annos_neg)
    labels = np.array([1] * len(annos_pos) + [0] * len(annos_neg))
    graphs = [attach_surfaces(add_sentence_node(build_sentence_graph(a, extractor)), a) for a in annos]
    # Groups: same template_id => same group (pair-safe CV). Items without one get their own group.
    groups = np.array([a["template_id"] if a.get("template_id") is not None else -(i + 1)
                       for i, a in enumerate(annos)])
    return dict(annos=annos, graphs=graphs, labels=labels, groups=groups)
