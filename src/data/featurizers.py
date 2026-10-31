"""
Shared frozen text encoders: SBERT, BERT (CLS), BERT-isolated, BERT-contextual.

Extracted from the RGAT notebook's version (notebookB_RGAT...), NOT the baselines notebook's,
because baselines' SBERTFeaturizer.node_key had a bug introduced when der_graph.py's node IDs
changed from plain strings to (span, node_type) tuples: it built the SBERT cache key from
``str(n)`` -- which on a tuple node ID produces the literal Python repr, e.g.
"('brahmin', 'Identity Dimensions')", and SBERT would embed THAT garbage string instead of
"brahmin". RGAT's version already used ``node_surface(G, n)`` (the clean original span text)
and does not have this bug.

Checked: this did NOT corrupt any already-reported baseline numbers, because baselines only
ever calls SBERT at the *sentence* level (MLP SBERT uses sentence_vector(), which takes the
Sentence-node branch of node_key and never hits the buggy line) and the LR bag-of-types/
bag-of-nodes/frames-triggers baselines use node_surface() directly via graph_features(), not
SBERT at all. BUT: the GAT/R-GCN notebook needs per-DER-node SBERT vectors (that's the whole
point of a DER-only graph classifier), and that notebook does not define its own featurizers --
it was written assuming FEATURIZERS already exists in the kernel from a prior notebook run in
the same session. If that prior run was baselines (pre-fix), its SBERT/GAT/R-GCN numbers are
built on garbage node embeddings and need rerunning. If it was RGAT, they're fine. Check your
run history; going forward, every notebook should build FEATURIZERS fresh via this module.

A .get(G, n) method is added directly to the Featurizer base class here, rather than via the
ad-hoc monkeypatch cells ("Fix: provide the missing get() method...") that existed in two of the
three notebooks -- same effect, no patch cell needed.
"""
import time
from collections import defaultdict
from typing import Dict

import numpy as np
import torch

from .der_graph import SENTENCE_NODE_TYPE


def sentence_node_name(G):
    return next(n for n, a in G.nodes(data=True) if a.get("node_type") == SENTENCE_NODE_TYPE)


def sentence_text(G):
    a = G.nodes[sentence_node_name(G)]
    return a.get("text", G.graph.get("sentence", ""))


def node_surface(G, n):  # original span text (falls back to the node name)
    return G.nodes[n].get("surface", str(n))


def node_ctx(G, n):  # sentence the node is contextualised in
    return G.nodes[n].get("ctx", G.graph.get("sentence", ""))


def find_char_span(sentence, span_text):
    """Case-insensitive, word-bounded span lookup (handles simple plurals)."""
    import re
    if not sentence or not span_text:
        return None
    cands = [span_text]
    if span_text.lower().endswith("y"):
        cands.append(span_text[:-1] + "ies")
    cands += [span_text + "s", span_text + "es"]
    for c in cands:
        m = re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", sentence, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    m = re.search(re.escape(span_text), sentence, flags=re.IGNORECASE)  # last resort
    return (m.start(), m.end()) if m else None


class Featurizer:
    """text/span -> vector cache. Keys are hashable; vectors live on CPU in fp32."""
    name, dim = "?", 0

    def __init__(self):
        self.cache, self.failures, self.released = {}, set(), False

    def node_key(self, G, n):
        raise NotImplementedError

    def _compute(self, missing):
        raise NotImplementedError

    def keys(self, G, nodes=None):
        nodes = list(G.nodes) if nodes is None else nodes
        return [self.node_key(G, n) for n in nodes]

    def ensure(self, keys):
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if not missing:
            return
        if self.released:
            raise RuntimeError(f"{self.name} encoder was released but {len(missing)} uncached keys were "
                               f"requested (e.g. {missing[0]}). Encode everything needed before release().")
        self._compute(missing)

    def warm(self, graphs):
        self.ensure([k for G in graphs for k in self.keys(G)])

    def vectors(self, G, nodes):
        ks = self.keys(G, nodes)
        self.ensure(ks)
        return torch.stack([self.cache[k] for k in ks])

    def sentence_vector(self, G):
        k = self.node_key(G, sentence_node_name(G))
        self.ensure([k])
        return self.cache[k]

    def get(self, G, n):
        """Single-node vector, used by the GAT/R-GCN/RGAT per-node pipelines."""
        k = self.node_key(G, n)
        self.ensure([k])
        return self.cache[k]


class SBERTFeaturizer(Featurizer):
    name = "SBERT"

    def __init__(self, device, encoder_fp16=True, enc_batch=64, model_name="all-MiniLM-L6-v2"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.device, self.enc_batch = device, enc_batch
        self.model = SentenceTransformer(model_name, device=str(device))
        if encoder_fp16 and device.type == "cuda":
            self.model.half()
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def node_key(self, G, n):
        a = G.nodes[n]
        if a.get("node_type") == SENTENCE_NODE_TYPE:
            return ("t", a.get("text", G.graph.get("sentence", "")))
        return ("t", node_surface(G, n))  # DER span text only -- NOT str(n); see module docstring

    def _compute(self, missing):
        texts = [k[1] for k in missing]
        with torch.no_grad():
            embs = self.model.encode(texts, batch_size=self.enc_batch, convert_to_tensor=True,
                                     show_progress_bar=False)
        embs = embs.float().cpu()
        for k, e in zip(missing, embs):
            self.cache[k] = e.clone()

    def release(self):
        self.model = None
        self.released = True


class BERTFeaturizer(Featurizer):
    name = "BERT"

    def __init__(self, device, encoder_fp16=True, enc_batch=64, model_name="bert-base-uncased"):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.device, self.enc_batch = device, enc_batch
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.dim = int(self.model.config.hidden_size)
        self.model_name = model_name
        self.fp16 = bool(encoder_fp16 and device.type == "cuda")

    def node_key(self, G, n):
        a = G.nodes[n]
        if a.get("node_type") == SENTENCE_NODE_TYPE:
            return ("cls", a.get("text", G.graph.get("sentence", "")))
        return ("span", node_ctx(G, n), node_surface(G, n))

    def _compute(self, missing):
        by_sent = defaultdict(list)
        for k in missing:
            by_sent[k[1]].append(k)
        sents = sorted(by_sent, key=len)  # length-sorted batches => little padding
        for b in range(0, len(sents), self.enc_batch):
            chunk = sents[b:b + self.enc_batch]
            enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True,
                           max_length=256, return_offsets_mapping=True)
            offsets = enc.pop("offset_mapping").numpy()
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                hidden = self.model(**enc).last_hidden_state
            hidden = hidden.float().cpu()
            for i, s in enumerate(chunk):
                for k in by_sent[s]:
                    if k[0] == "cls":
                        vec = hidden[i, 0]
                    else:
                        span = find_char_span(s, k[2])
                        ids = [] if span is None else [j for j, (a, c) in enumerate(offsets[i])
                                                       if c > a and a < span[1] and c > span[0]]
                        if ids:
                            vec = hidden[i, ids].mean(dim=0)
                        else:
                            vec = torch.zeros(self.dim)
                            self.failures.add((s, k[2]))
                    self.cache[k] = vec.clone()
            del hidden, enc

    def release(self):
        self.model = None
        self.tok = None
        self.released = True


class BERTIsolatedFeaturizer(BERTFeaturizer):
    """BERT CLS vectors for each DER node's text in isolation; sentence nodes stay contextual CLS."""
    name = "BERT-ISO"

    def node_key(self, G, n):
        a = G.nodes[n]
        if a.get("node_type") == SENTENCE_NODE_TYPE:
            return ("cls", a.get("text", G.graph.get("sentence", "")))
        return ("isolated", node_surface(G, n))

    def _compute(self, missing):
        sentence_keys = [k for k in missing if k[0] == "cls"]
        isolated_keys = [k for k in missing if k[0] == "isolated"]
        for keys, texts in [(sentence_keys, [k[1] for k in sentence_keys]),
                            (isolated_keys, [k[1] for k in isolated_keys])]:
            if not keys:
                continue
            for start in range(0, len(keys), self.enc_batch):
                key_chunk = keys[start:start + self.enc_batch]
                text_chunk = texts[start:start + self.enc_batch]
                enc = self.tok(text_chunk, return_tensors="pt", padding=True, truncation=True, max_length=256)
                enc = {k: v.to(self.device) for k, v in enc.items()}
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                    hidden = self.model(**enc).last_hidden_state[:, 0, :]
                hidden = hidden.float().cpu()
                for k, vec in zip(key_chunk, hidden):
                    self.cache[k] = vec.clone()
                del hidden, enc


class BERTContextFeaturizer:
    """Mean token embedding for each annotated span in its full sentence context.

    Deliberately a standalone class (not a Featurizer subclass) -- this is how it was written in
    the RGAT notebook and changing its shape risks changing its cached-key behaviour. It exposes
    the same .get(G, n) surface as Featurizer so calling code doesn't need to care which it has.
    """
    dim = 768

    def __init__(self, device, encoder_fp16=True):
        from transformers import AutoTokenizer, AutoModel
        self.device, self.fp16 = device, bool(encoder_fp16 and device.type == "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        self.model = AutoModel.from_pretrained("bert-base-uncased").to(device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.cache = {}

    def node_key(self, G, node):
        attrs = G.nodes[node]
        return (G.graph.get("sentence", ""), attrs.get("node_type", "Unknown"), attrs.get("surface", str(node)))

    @torch.no_grad()
    def _encode(self, sentence, spans):
        enc = self.tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True,
                             truncation=True, max_length=256)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            hidden = self.model(**enc).last_hidden_state[0].float().cpu()
        out = {}
        for span in spans:
            start = sentence.lower().find(span.lower())
            if start < 0:
                out[span] = np.zeros(self.dim, dtype=np.float32)
                continue
            end = start + len(span)
            ids = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
            out[span] = (hidden[ids].mean(0).numpy().astype(np.float32) if ids
                         else np.zeros(self.dim, dtype=np.float32))
        return out

    def ensure(self, keys):
        by_sentence = {}
        for sentence, node_type, span in keys:
            if (sentence, node_type, span) not in self.cache:
                by_sentence.setdefault(sentence, set()).add(span)
        for sentence, spans in by_sentence.items():
            encoded = self._encode(sentence, sorted(spans))
            for span in spans:
                self.cache[(sentence, None, span)] = encoded[span]
        for sentence, node_type, span in keys:
            self.cache[(sentence, node_type, span)] = self.cache[(sentence, None, span)]

    def get(self, G, node):
        return torch.from_numpy(self.cache[self.node_key(G, node)])

    def release(self):
        self.model.to("cpu")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_featurizers(device, data, sentence_node_type=SENTENCE_NODE_TYPE, encoder_fp16=True,
                      enc_batch=64, which=("SBERT", "BERT", "BERT-ISO", "BERT-CTX")):
    """Build every requested featurizer fresh and warm its cache over `data` (a dict of
    {dataset_name: {"graphs": [...]}} as produced by der_graph.build_dataset), so every notebook
    gets FEATURIZERS from the SAME code with no dependency on another notebook's kernel state.
    """
    out: Dict[str, object] = {}
    all_graphs = [G for d in data.values() for G in d["graphs"]]

    if "SBERT" in which:
        t0 = time.time()
        out["SBERT"] = SBERTFeaturizer(device, encoder_fp16, enc_batch)
        out["SBERT"].warm(all_graphs)
        print(f"SBERT: dim={out['SBERT'].dim}, {len(out['SBERT'].cache)} cached vectors, {time.time()-t0:.0f}s")

    if "BERT" in which:
        t0 = time.time()
        out["BERT"] = BERTFeaturizer(device, encoder_fp16, enc_batch)
        out["BERT"].warm(all_graphs)
        print(f"BERT: dim={out['BERT'].dim}, {len(out['BERT'].cache)} cached vectors, {time.time()-t0:.0f}s")

    if "BERT-ISO" in which:
        t0 = time.time()
        feat = BERTIsolatedFeaturizer(device, encoder_fp16, enc_batch)
        feat.warm(all_graphs)
        out["BERT-ISO"] = feat
        print(f"BERT-ISO: {len(feat.cache)} cached vectors, {time.time()-t0:.0f}s")

    if "BERT-CTX" in which:
        t0 = time.time()
        feat = BERTContextFeaturizer(device, encoder_fp16)
        keys = [feat.node_key(G, n) for G in all_graphs for n, a in G.nodes(data=True)
               if a.get("node_type") != sentence_node_type]
        feat.ensure(keys)
        out["BERT-CTX"] = feat
        print(f"BERT-CTX: cached {len(feat.cache)} DER-node vectors in {time.time()-t0:.0f}s")

    return out
