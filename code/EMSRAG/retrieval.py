# retrieval.py

import os
import json
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from typing import List, Optional
from tqdm import tqdm
import torch
import torch.nn.functional as F
import multiprocessing
from multiprocessing import Pool
import argparse
import time

class EMSRetriever:
    def __init__(
        self,
        kb_data_dir: str,
        pr_data_dir: str,
        use_kb: bool,
        use_pr: bool,
        article_model_name: str = "ncbi/MedCPT-Article-Encoder",
        query_model_name: str   = "ncbi/MedCPT-Query-Encoder",

        kb_index_path: str         = "faiss.index",
        kb_mapping_path: str       = "id_mapping.jsonl",

        pr_index_path: str      = "pr.index",
        pr_mapping_path: str    = "pr_mapping.jsonl",
        pr_files: Optional[List[str]] = None,
        chunk_strategy: str     = "token",
        token_chunk_size: int   = 512,
        token_overlap: int      = 128,
        index_type: str         = "flat",
        hnsw_m: int             = 32,
        ivf_nlist: int          = 100
    ):
        self.use_kb = use_kb
        self.use_pr = use_pr
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.use_kb:
            # config & paths
            self.kb_data_dir        = kb_data_dir
            self.kb_emb_dir         = os.path.join(self.kb_data_dir, "embeddings")
            os.makedirs(self.kb_emb_dir, exist_ok=True)
            self.kb_index_path      = os.path.join(self.kb_emb_dir, kb_index_path)
            self.kb_mapping_path    = os.path.join(self.kb_emb_dir, kb_mapping_path)

        # ----- Patient‐Record config & state -----
        if self.use_pr:
            self.pr_data_dir     = pr_data_dir
            self.pr_files = pr_files
            self.pr_emb_dir      = os.path.join(self.pr_data_dir, "embeddings")
            os.makedirs(self.pr_emb_dir, exist_ok=True)
            self.pr_index_path   = os.path.join(self.pr_emb_dir, pr_index_path)
            self.pr_mapping_path = os.path.join(self.pr_emb_dir, pr_mapping_path)


        # ----- common settings -----
        self.chunk_strategy  = chunk_strategy
        self.token_chunk_size= token_chunk_size
        self.token_overlap   = token_overlap
        self.index_type      = index_type.lower()
        self.hnsw_m          = hnsw_m
        self.ivf_nlist       = ivf_nlist

        # document-side encoder + tokenizer
        self.doc_model      = SentenceTransformer(article_model_name)
        self.doc_tokenizer  = AutoTokenizer.from_pretrained(article_model_name, use_fast=True)
        # query-side encoder (truncate at 64 tokens)
        self.query_model    = SentenceTransformer(query_model_name)
        self.query_model.tokenizer.model_max_length = 64
        self.query_model.tokenizer.init_kwargs["truncation"] = True

        # # in‐memory KB maps
        if self.use_kb:
            self.kb_id2doc        = {}   # doc_id → {title,content}
            self.kb_id_list       = []   # global ordered list of doc_ids
            self.kb_category  = {}   # doc_id → category stem
            self.kb_cat_indexes = {}    # cat → FAISS index
            self.kb_cat_idlists = {}    # cat → [doc_id_0, doc_id_1, …]
            self.kb_index  = None  # FAISS global index
            self.kb_cats_map = {
                "anatomy": "anatomy",
                "others": "others",
                "pediatrics": "pediatrics",
                "ems_operations": "ems_operations",
                "cardiology_and_resuscitation": "cardiovascular",
                "trauma": "trauma",
                "medical_and_obstetrics_gynecology": "medical",
                "airway_respiration_and_ventilation": "airway, respiratory, ventilation",
                "assessment": "assessment",
                "pharmacology": "pharmacology"
            }


        # in‐memory PR maps
        if self.use_pr:
            self.pr_id2doc      = {}
            self.pr_id_list     = []
            self.pr_category    = {}
            self.pr_index       = None
            self.pr_cat_indexes = {}   # cat -> faiss.Index
            self.pr_cat_idlists = {}   # cat -> [doc_id0, doc_id1, …]
            self.pr_cats_map = {
                "pediatrics": "pediatrics",
                "ems_operations": "ems operations",
                "cardiology_and_resuscitation": "cardiovascular",
                "trauma": "trauma",
                "medical_and_obstetrics_gynecology": "medical & ob",
                "airway_respiration_and_ventilation": "airway",
                "assessment": "assessment"
            }

        if self.use_kb:
            if (
                os.path.exists(self.kb_index_path)
                and os.path.exists(self.kb_mapping_path)
            ):

                # with open(self.kb_mapping_path, "r") as f:
                #     mp = json.load(f)
                # self.kb_id2doc, self.kb_id_list, self.kb_category = (
                #     mp["id2doc"], mp["id_list"], mp["doc_category"]
                # )

                with open(self.kb_mapping_path, "r") as f:
                    for line in f:
                        rec = json.loads(line)
                        did = rec["id"]
                        self.kb_id_list.append(did)
                        self.kb_category[did] = rec["category"]
                        self.kb_id2doc[did] = {"title":   rec["title"], "content": rec["content"]}


                self.kb_index = faiss.read_index(self.kb_index_path)
                for cat in set(self.kb_category.values()):
                    idx_path = os.path.join(self.kb_emb_dir, f"{cat}.index")
                    self.kb_cat_indexes[cat] = faiss.read_index(idx_path)
                    # rebuild the local‐id→doc_id list from your global lists:
                    self.kb_cat_idlists[cat] = [
                        did for did in self.kb_id_list
                        if self.kb_category[did] == cat
                    ]

            else:
                self._build_kb_index()
        
        if self.use_pr:
            if (
                os.path.exists(self.pr_index_path)
                and os.path.exists(self.pr_mapping_path)
            ):
                with open(self.pr_mapping_path, 'r') as f:
                    for line in f:
                        rec = json.loads(line)
                        did = rec['id']
                        self.pr_id2doc[did] = {'category': rec['category'], 'text': rec['text']}
                        self.pr_category[did] = rec['category']
                        self.pr_id_list.append(did)
                self.pr_index = faiss.read_index(self.pr_index_path)

                # load per-category PR indexes & id‐lists
                for cat in set(self.pr_category.values()):
                    idx_path = os.path.join(self.pr_emb_dir, f"{cat}_pr.index")
                    if os.path.exists(idx_path):
                        # 1) read the small index
                        idx = faiss.read_index(idx_path)
                        self.pr_cat_indexes[cat] = idx
                        # 2) rebuild the same order of doc_ids you wrote out in mapping
                        #    (you already have a global pr_id_list and pr_category)
                        self.pr_cat_idlists[cat] = [
                            pid for pid in self.pr_id_list
                            if self.pr_category[pid] == cat
                        ]

            else:
                self._build_pr_index()

    def _chunk_content(self, content: str):
        if self.chunk_strategy == "none":
            return [content]
        if self.chunk_strategy == "paragraph":
            return [p.strip() for p in content.split("\n\n") if p.strip()]

        sentences = re.split(r'(?<=[.?!])\s+', content)
        if self.chunk_strategy == "sentence":
            return sentences

        # token-based sliding window
        tokens = self.doc_tokenizer.encode(content, add_special_tokens=False)
        stride = self.token_chunk_size - self.token_overlap
        chunks = []
        for i in range(0, len(tokens), stride):
            piece = tokens[i : i + self.token_chunk_size]
            txt   = self.doc_tokenizer.decode(piece,
                                              skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
            if txt.strip():
                chunks.append(txt)
        return chunks

    def _extract_chunks(self, text: str) -> list:
        """
        Chunk a single text string via tokenizer overflow.
        Returns list of string chunks.
        """
        win = self.token_chunk_size
        stride = win - self.token_overlap
        enc = self.doc_tokenizer(
            text,
            max_length=win,
            stride=stride,
            truncation=True,
            return_overflowing_tokens=True
        )
        chunks = []
        for token_ids in enc['input_ids']:
            chunk = self.doc_tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            chunks.append(chunk)
        return chunks

    def _build_kb_index(self):
        all_embs = []
        # iterate every JSON file in data_dir
        for fname in sorted(os.listdir(self.kb_data_dir)):
            if not fname.endswith(".json"):
                continue
            stem  = fname.rsplit(".",1)[0]
            pages = json.load(open(os.path.join(self.kb_data_dir, fname)))

            texts, ids, metas = [], [], []
            for p_idx, page in enumerate(pages):
                for e_idx, (title, content) in enumerate(page.items()):
                    chunks = self._chunk_content(content)
                    for c_idx, chunk in enumerate(chunks):
                        doc_id = f"{stem}_{p_idx}_{e_idx}" if len(chunks)==1 else f"{stem}_{p_idx}_{e_idx}_{c_idx}"
                        texts.append(f"{title}. {chunk}")
                        ids.append(doc_id)
                        metas.append({"id":doc_id, "category":stem,
                                    "title": title, "content": chunk})

                        self.kb_id2doc[doc_id]      = {"title": title, "content": chunk}
                        self.kb_category[doc_id] = stem

            # embed & normalize
            emb = self.doc_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            faiss.normalize_L2(emb)
            all_embs.append(emb)
            np.save(os.path.join(self.kb_emb_dir, f"{stem}_embeddings.npy"), emb)
            self.kb_id_list.extend(ids)

            # --- build & save a per-stem FAISS index ---
            dim = emb.shape[1]
            if self.index_type == "flat":
                cidx = faiss.IndexFlatIP(dim)
            elif self.index_type == "hnsw":
                cidx = faiss.IndexHNSWFlat(dim, self.hnsw_m)
                cidx.metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                quant = faiss.IndexFlatIP(dim)
                cidx = faiss.IndexIVFFlat(quant, dim, self.ivf_nlist,
                                        faiss.METRIC_INNER_PRODUCT)
                cidx.train(emb)
            cidx.add(emb)
            idx_path = os.path.join(self.kb_emb_dir, f"{stem}.index")
            faiss.write_index(cidx, idx_path)
            # --- save per-stem mapping JSONL ---
            map_path = os.path.join(self.kb_emb_dir, f"{stem}_map.jsonl")
            with open(map_path, "w") as mf:
                for m in metas:
                    mf.write(json.dumps(m, ensure_ascii=False) + "\n")

        # stack, normalize, save global embeddings
        global_emb = np.vstack(all_embs)
        faiss.normalize_L2(global_emb)
        np.save(os.path.join(self.kb_emb_dir, "embeddings.npy"), global_emb)

        # build global FAISS index
        dim = global_emb.shape[1]
        if self.index_type == "flat":
            idx = faiss.IndexFlatIP(dim)
        elif self.index_type == "hnsw":
            idx = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            idx.metric_type = faiss.METRIC_INNER_PRODUCT
        else:  # ivf
            quant = faiss.IndexFlatIP(dim)
            idx = faiss.IndexIVFFlat(quant, dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(global_emb)

        idx.add(global_emb)
        faiss.write_index(idx, self.kb_index_path)
        self.kb_index = idx

        # save the global mapping as JSONL
        with open(self.kb_mapping_path, "w") as gf:
            for did in self.kb_id_list:
                rec = {
                    "id":       did,
                    "category": self.kb_category[did],
                    "title":    self.kb_id2doc[did]["title"],
                    "content":  self.kb_id2doc[did]["content"]
                }
                gf.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _build_pr_index(self):
        """
        Build per-category patient-record index without multiprocessing.
        Saves per-category embeddings and mapping, then global index.
        """
        if not self.pr_files:
            pr_files = [
                f for f in sorted(os.listdir(self.pr_data_dir))
                if f.endswith('.jsonl') and f.lower() != 'nemsis.jsonl'
            ]
        else:
            pr_files = self.pr_files

        # initialize global containers
        self.pr_id2doc = {}
        self.pr_category = {}
        self.pr_id_list = []
        batch_size = 2048
        all_embs = []

        for fname in tqdm(pr_files, desc='Categories', unit='file'):
            cat = fname.replace(".jsonl", "")
            emb_path = os.path.join(self.pr_emb_dir, f"{cat}_pr_embeddings.npy")
            map_path = os.path.join(self.pr_emb_dir, f"{cat}_pr_map.jsonl")
            idx_path = os.path.join(self.pr_emb_dir, f"{cat}_pr.index")

            # reload caches
            if os.path.exists(emb_path) and os.path.exists(map_path) and os.path.exists(idx_path):
                emb = np.load(emb_path)
                faiss.normalize_L2(emb)
                all_embs.append(emb)

                # load mapping
                with open(map_path,'r') as mf:
                    for line in mf:
                        rec = json.loads(line)
                        did = rec['id']
                        self.pr_id2doc[did] = {'category':rec['category'],'text':rec['text']}
                        self.pr_category[did] = rec['category']
                        self.pr_id_list.append(did)
                continue

            # chunk and embed
            texts, ids = [], []
            with open(os.path.join(self.pr_data_dir,fname),'r') as rf, open(map_path,'w') as mf:
                for line in tqdm(rf, desc=f"Chunking {cat}", unit='rec', leave=False):
                    for raw_id, txt in json.loads(line).items():
                        for i, chunk in enumerate(self._extract_chunks(txt)):
                            did = f"PR_{cat}_{raw_id}" if i==0 and len(self._extract_chunks(txt))==1 else f"PR_{cat}_{raw_id}_{i}"
                            texts.append(chunk)
                            ids.append(did)
                            # write mapping record line-by-line
                            mf.write(json.dumps({'id': did, 'category': cat, 'text': chunk}) + "\n")

            # embed in batches
            emb_slices=[]
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                tensor = self.doc_model.encode(
                    batch,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                tensor = F.normalize(tensor,p=2,dim=1)
                emb_slices.append(tensor.cpu().numpy())
            emb = np.vstack(emb_slices)
            faiss.normalize_L2(emb)
            np.save(emb_path, emb)

            all_embs.append(emb)
            self.pr_id_list.extend(ids)

            # build & save per-cat FAISS index
            dim = emb.shape[1]
            if self.index_type=='flat':
                cidx = faiss.IndexFlatIP(dim)
            elif self.index_type=='hnsw':
                cidx = faiss.IndexHNSWFlat(dim,self.hnsw_m); cidx.metric_type=faiss.METRIC_INNER_PRODUCT
            else:
                quant = faiss.IndexFlatIP(dim)
                cidx = faiss.IndexIVFFlat(quant,dim,self.ivf_nlist,faiss.METRIC_INNER_PRODUCT); cidx.train(emb)
            cidx.add(emb)
            faiss.write_index(cidx, idx_path)

        # build global embeddings and index
        if not all_embs:
            raise RuntimeError('No patient-record embeddings built')
        global_emb = np.vstack(all_embs)
        faiss.normalize_L2(global_emb)
        np.save(os.path.join(self.pr_emb_dir, 'pr_embeddings.npy'), global_emb)

        dim = global_emb.shape[1]
        if self.index_type == 'flat':
            idx = faiss.IndexFlatIP(dim)
        elif self.index_type == 'hnsw':
            idx = faiss.IndexHNSWFlat(dim, self.hnsw_m)
            idx.metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            quant = faiss.IndexFlatIP(dim)
            idx = faiss.IndexIVFFlat(quant, dim, self.ivf_nlist, faiss.METRIC_INNER_PRODUCT)
            idx.train(global_emb)
        idx.add(global_emb)
        # faiss.write_index(idx, self.pr_mapping_path.replace('map.json', 'index'))
        faiss.write_index(idx, os.path.join(self.pr_emb_dir,'pr.index'))
        self.pr_index = idx

        # save global mapping JSONL
        with open(self.pr_mapping_path,'w') as gf:
            for did in self.pr_id_list:
                gf.write(json.dumps({'id':did,'category':self.pr_category[did],'text':self.pr_id2doc[did]['text']})+"\n")

    def retrieve(
        self,
        query: str,
        kb_k:  int = 5,
        pr_k:  int = 0,
        categories: Optional[List[str]] = None,
        mode: str = "global",
    ):
        """
        query → embed (≤64 tokens) → search.
        """

        if self.use_kb and kb_k <= 0:
            raise Exception("kb_k >= 0 when set use_kb=True")

        if self.use_pr and pr_k <=0:
            raise Exception("pr_k >= 0 when set use_pr=True")

        # normalize categories list
        cats = None if (not categories or "all" in categories) else {c.lower() for c in categories}
        q_emb = self.query_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        kb_out = []
        pr_out = []

        # --- KB side ---
        if self.use_kb:
            if cats:
                kb_cats = [self.kb_cats_map.get(c) for c in cats]
            if mode == "global":
                scores, idxs = self.kb_index.search(q_emb, kb_k)
                for i, idx in enumerate(idxs[0]):
                    kb_out.append({
                        "id": self.kb_id_list[idx],
                        **self.kb_id2doc[self.kb_id_list[idx]],
                        "score": float(scores[0][i])
                    })
            
            elif mode == "retrieve_then_filter":
                # --- option 1: filter global ---
                # oversample so you have enough after filter, all embeddings -> filter based on category
                scores, idxs = self.kb_index.search(q_emb, kb_k * 10)
                for sc, idx in zip(scores[0], idxs[0]):
                    doc_id = self.kb_id_list[idx]
                    if self.kb_category[doc_id] in kb_cats:
                        meta = self.kb_id2doc[doc_id]
                        kb_out.append({"id":doc_id, **meta, "score":float(sc)})
                        if len(kb_out) == kb_k:
                            break

            elif mode == "filter_then_retrieve":
                all_hits = []

                # for each requested category, query its small index
                for cat in kb_cats:
                    idx = self.kb_cat_indexes.get(cat)
                    local_ids = self.kb_cat_idlists[cat]
                    # ask for up to kb_k candidates per category
                    scores, idxs = idx.search(q_emb, kb_k)

                    for score, local_i in zip(scores[0], idxs[0]):
                        doc_id = local_ids[local_i]
                        meta   = self.kb_id2doc[doc_id]
                        all_hits.append({
                            "id":    doc_id,
                            **meta,
                            "score": float(score)
                        })

                # now merge, sort by score, and take the top-kb_k unique
                all_hits.sort(key=lambda x: x["score"], reverse=True)
                seen = set()
                kb_out = []
                for hit in all_hits:
                    if hit["id"] not in seen:
                        seen.add(hit["id"])
                        kb_out.append(hit)
                        if len(kb_out) == kb_k:
                            break

            else:
                raise ValueError("mode must be 'global' or 'retrieve_then_filter' or 'filter_then_retrieve'")

        # --- PR side ---
        if self.use_pr:
            if cats:
                pr_cats = [self.pr_cats_map[c] for c in cats if c in self.pr_cats_map]
                if not pr_cats:
                    return {"kb": kb_out, "pr": []}


            if mode == "global":
                scores, idxs = self.pr_index.search(q_emb, pr_k)
                for rank, idx in enumerate(idxs[0]):
                    pid = self.pr_id_list[idx]
                    pr_out.append({"id":pid, **self.pr_id2doc[pid], "score":float(scores[0][rank])})

            elif mode == "retrieve_then_filter":
                scores, idxs = self.pr_index.search(q_emb, pr_k*10)
                for sc, idx in zip(scores[0], idxs[0]):
                    pid = self.pr_id_list[idx]
                    if self.pr_category[pid] in pr_cats:
                        pr_out.append({"id":pid, **self.pr_id2doc[pid], "score":float(sc)})
                        if len(pr_out) >= pr_k: break

            elif mode == "filter_then_retrieve":
                # for each requested PR category, search its small index
                all_pr_hits = []
                for cat in pr_cats:
                    idx = self.pr_cat_indexes[cat]

                    local_ids = self.pr_cat_idlists[cat]
                    # ask for up to pr_k per category
                    scores, idxs = idx.search(q_emb, pr_k)
                    for sc, loc_i in zip(scores[0], idxs[0]):
                        pid = local_ids[loc_i]
                        rec = self.pr_id2doc[pid]
                        all_pr_hits.append({
                            "id":    pid,
                            "text":  rec["text"],
                            "category": rec["category"],
                            "score": float(sc)
                        })

                # global re-rank + dedupe
                all_pr_hits.sort(key=lambda x: x["score"], reverse=True)
                seen = set()
                for hit in all_pr_hits:
                    if hit["id"] not in seen:
                        seen.add(hit["id"])
                        pr_out.append(hit)
                        if len(pr_out) == pr_k:
                            break

            else:
                raise ValueError("mode must be 'global' or 'retrieve_then_filter' or 'filter_then_retrieve'")
    
        return {"kb": kb_out, "pr": pr_out}



def evaluate_semantic_coverage(
    qa_list,                   # list of dicts, each has a "question" field
    retriever,                 # your EMSRetriever instance
    kb_k=8,                   # FAISS top-k for KB
    pr_k=8,                   # FAISS top-k for PR
    mode="global",             # "global"/"filter"/"subset"
    log_dir=None               # directory to cache per-question results
):
    """
    For each question i:
      - If log_dir/i.json exists, load { "max_kb":…, "max_pr":… }
      - Else: call retriever.retrieve, extract max sim, write cache file.
    Returns overall:
      { "KB_Hit@τ":…, "KB_MeanMax":…, "PR_Hit@τ":…, "PR_MeanMax":… }
    """
    # prepare cache directory
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    per_q_results = []
    for idx, qa in enumerate(tqdm(qa_list, desc="Eval")):
        cache_path = os.path.join(log_dir, f"{idx}.json") if log_dir else None

        if cache_path and os.path.exists(cache_path):
            # load cached
            with open(cache_path, 'r') as f:
                res = json.load(f)
        else:
            # compute
            out = retriever.retrieve(
                query=qa["question"],
                kb_k=kb_k if retriever.use_kb else 0,
                pr_k=pr_k if retriever.use_pr else 0,
                categories=qa["category"] if mode!="global" else "all",
                mode=mode
            )

            res = {}
            if retriever.use_kb:
                sims_kb = [d["score"] for d in out["kb"]]
                res["max_kb"] = max(sims_kb) if sims_kb else 0.0

            if retriever.use_pr:
                sims_pr = [d["score"] for d in out["pr"]]
                res["max_pr"] = max(sims_pr) if sims_pr else 0.0

            # write cache
            if cache_path:
                with open(cache_path, 'w') as f:
                    json.dump(res, f)

        per_q_results.append(res)

    with open(os.path.join(log_dir, "overall.json"), "w") as f:
        json.dump(per_q_results, f, indent=4)

    # aggregate metrics
    n = len(per_q_results)
    metrics = {}

    if retriever.use_kb:
        kb_maxes = [r.get("max_kb", 0.0) for r in per_q_results]
        for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
            kb_hits   = sum(1 for s in kb_maxes if s >= tau)
            metrics[f"KB_Hit@{tau}"]   = kb_hits / n

        metrics["KB_MeanMax"] = float(np.mean(kb_maxes))

    if retriever.use_pr:
        pr_maxes = [r.get("max_pr", 0.0) for r in per_q_results]
        for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
            pr_hits   = sum(1 for s in pr_maxes if s >= tau)
            metrics[f"PR_Hit@{tau}"]   = pr_hits / n
        metrics["PR_MeanMax"] = float(np.mean(pr_maxes))

    return metrics




if __name__ == "__main__":
    ################################################## only Kb #################################################
    # retriever = EMSRetriever(
    #     kb_data_dir="/scratch/zar8jw/EMS-MCQA/knowledge",
    #     pr_data_dir="/scratch/zar8jw/NEMSIS/csv",
    #     use_kb=True,
    #     use_pr=False
    # )

    # query = "the term \u201cmoderate burns\u201d describes a certain degree of burn covering a certain percentage of the body (using the rule of nines for adults). which of the following does not fall under the moderate burns category for adult patients?"
    # cats  = ["trauma"]

    # print(f"→ Global")
    # for doc in retriever.retrieve(query=query, kb_k=10, pr_k=0, mode="global")["kb"]:
    #     print(f"[{doc['score']:.3f}] {doc['title']} ({doc['id'].split('_')[0]})")

    # print("\n→ Retrieve then filter:")
    # for doc in retriever.retrieve(query, kb_k=10, pr_k=0, categories=cats, mode="retrieve_then_filter")["kb"]:
    #     print(f"[{doc['score']:.3f}] {doc['title']} ({doc['id'].split('_')[0]})")

    # print("\n→ filter then retrieve:")
    # for doc in retriever.retrieve(query, kb_k=10, pr_k=0, categories=cats, mode="filter_then_retrieve")["kb"]:
    #     print(f"[{doc['score']:.3f}] {doc['title']} ({doc['id'].split('_')[0]})")


    ################################################# only PC #################################################
    # start = time.time()
    # retriever = EMSRetriever(
    #     kb_data_dir="/scratch/zar8jw/EMS-MCQA/knowledge",
    #     pr_data_dir="/scratch/zar8jw/NEMSIS/csv",
    #     use_kb=False,
    #     use_pr=True
    # )
    # end = time.time()
    # print(f"time to initilization: {end-start}")

    # query = "the term \u201cmoderate burns\u201d describes a certain degree of burn covering a certain percentage of the body (using the rule of nines for adults). which of the following does not fall under the moderate burns category for adult patients?"
    # cats  = ["medical_and_obstetrics_gynecology"]

    # print(f"→ Global")
    # for doc in retriever.retrieve(query=query, kb_k=0, pr_k=3, mode="global")["pr"]:
    #     print(f"[{doc['score']:.3f}] ({doc['id']}) {doc['text']}")

    # print("\n→ Retrieve then filter")
    # for doc in retriever.retrieve(query, kb_k=0, pr_k=3, categories=cats, mode="retrieve_then_filter")["pr"]:
    #     print(f"[{doc['score']:.3f}] ({doc['id']}) {doc['text']}")

    # print("\n→ Filter then retrieve")
    # for doc in retriever.retrieve(query, kb_k=0, pr_k=3, categories=cats, mode="filter_then_retrieve")["pr"]:
    #     print(f"[{doc['score']:.3f}] ({doc['id']}) {doc['text']}")



    ################################################# Both KB and PC #################################################
    # start = time.time()
    # retriever = EMSRetriever(
    #     kb_data_dir="/scratch/zar8jw/EMS-MCQA/knowledge",
    #     pr_data_dir="/scratch/zar8jw/NEMSIS/csv",
    #     use_kb=True,
    #     use_pr=True
    # )
    # end = time.time()
    # print(f"time to initilization: {end-start}")

    # query = "the term \u201cmoderate burns\u201d describes a certain degree of burn covering a certain percentage of the body (using the rule of nines for adults). which of the following does not fall under the moderate burns category for adult patients?"
    # cats  = ["medical_and_obstetrics_gynecology"]

    # print(f"→ Global")
    # doc = retriever.retrieve(query=query, kb_k=1, pr_k=1, mode="global")
    # kb, pr = doc["kb"], doc["pr"]
    # for i in range(len(kb)):
    #     print(f"[{kb[i]['score']:.3f}] ({kb[i]['id']}) {kb[i]['content']}")
    # for j in range(len(pr)):
    #     print(f"[{pr[j]['score']:.3f}] ({pr[j]['id']}) {pr[j]['text']}")

    # print("\n→ Retrieve then Filter:")
    # doc = retriever.retrieve(query, kb_k=1, pr_k=1, categories=cats, mode="retrieve_then_filter")
    # kb, pr = doc["kb"], doc["pr"]
    # for i in range(len(kb)):
    #     print(f"[{kb[i]['score']:.3f}] ({kb[i]['id']}) {kb[i]['content']}")
    # for j in range(len(pr)):
    #     print(f"[{pr[j]['score']:.3f}] ({pr[j]['id']}) {pr[j]['text']}")

    # print("\n→ Filter then Retrieve:")
    # doc = retriever.retrieve(query, kb_k=1, pr_k=1, categories=cats, mode="filter_then_retrieve")
    # kb, pr = doc["kb"], doc["pr"]
    # for i in range(len(kb)):
    #     print(f"[{kb[i]['score']:.3f}] ({kb[i]['id']}) {kb[i]['content']}")
    # for j in range(len(pr)):
    #     print(f"[{pr[j]['score']:.3f}] ({pr[j]['id']}) {pr[j]['text']}")


    ################################################# calculate the similarity metrics #################################################

    parser = argparse.ArgumentParser(description="Evaluate EMS MCQA with the LLM of your choice")
    parser.add_argument(
        "--filter_mode",
        choices=["retrieve_then_filter", "filter_then_retrieve", "global"],
        type=str,
        default="retrieve_then_filter",
        help="filtering mode"
    )
    args = parser.parse_args()
    mode_name = args.filter_mode

    # 1) init retriever
    retriever = EMSRetriever(
        kb_data_dir="/scratch/zar8jw/EMS-MCQA/knowledge",
        pr_data_dir="/scratch/zar8jw/NEMSIS/csv",
        use_kb=True,
        use_pr=True
    )

    for src in ["open", "close"]:
        qa_list = json.load(open(f"/scratch/zar8jw/EMS-MCQA/data/final/MCQA_{src}_final.json"))   # list of dicts

        if mode_name == "global":
            # 2) run evaluation with caching
            metrics = evaluate_semantic_coverage(
                qa_list=qa_list,
                retriever=retriever,
                kb_k=5, 
                pr_k=5,
                mode=mode_name,
                log_dir=f"/scratch/zar8jw/EMS-MCQA/log/semantic_eval/{src}/{mode_name}"
            )
        elif mode_name == "filter_then_retrieve":
                        # 2) run evaluation with caching
            metrics = evaluate_semantic_coverage(
                qa_list=qa_list,
                retriever=retriever,
                kb_k=5, 
                pr_k=5,
                mode=mode_name,
                log_dir=f"/scratch/zar8jw/EMS-MCQA/log/semantic_eval/{src}/{mode_name}"
            )

        print(metrics)
        with open(os.path.join(f"/scratch/zar8jw/EMS-MCQA/log/semantic_eval/{src}", f"metrics_{mode_name}.json"), "w") as f:
            json.dump(metrics, f, indent=4)