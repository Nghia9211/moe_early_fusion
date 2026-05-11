"""
moe_rec_agent.py — Patched: Phase 1+2+3
  - Phase 1: Reranker bị bypass hoàn toàn (dùng pass-through embed_only
             chỉ để build rerank_scores shape, không sort lại c_m)
  - Phase 2: FeedbackScoreAdjuster inject vào fused_scores sau MoE fusion
  - Phase 3: update_memory() cập nhật FeedbackAdjuster sau mỗi round reject
"""

import os
import sys
import threading
import traceback
import math
from typing import Dict, List, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config              import MoEConfig, get_config_for_dataset, DEFAULT_CONFIG
from seq_scorer          import SeqScorer
from gcn_scorer          import GCNScorer
from semantic_scorer     import SemanticScorer
from candidate_retriever import CandidateRetriever
from gating_network      import GatingNetwork
from moe_fusion          import MoEFusion
from reranker            import Reranker, rank_to_score
from score_combiner      import ScoreCombiner, moe_confidence_score
from feedback_adjuster   import FeedbackScoreAdjuster          # ← Phase 2


class MoERecAgent:
    _shared_embedding      = None
    _shared_vector_store   = None
    _shared_gcn_embeddings = None
    _shared_sasrec_model   = None
    _shared_id2name:  Dict[int, str] = None
    _shared_name2id:  Dict[str, int] = None
    _shared_id2rawid: Dict[int, str] = None
    _shared_seq_size       = None
    _shared_item_num       = None
    _shared_device         = None
    _init_lock = threading.Lock()

    def __init__(self, args, moe_config=None, item_name_map=None, llm=None):
        self.args          = args
        self.memory:       List[str]  = []
        self.info_list:    List[dict] = []
        self.item_name_map = item_name_map or {}
        self.llm           = llm
        self.dataset = next(
            (d for d in ['yelp', 'amazon', 'goodreads'] if d in getattr(args, 'data_dir', '')),
            'amazon'
        )
        self.cfg = get_config_for_dataset(dataset=self.dataset)

        with MoERecAgent._init_lock:
            self._init_shared_resources(args)

        self.sasrec_model   = MoERecAgent._shared_sasrec_model
        self.gcn_embeddings = MoERecAgent._shared_gcn_embeddings
        self.embedding_fn   = MoERecAgent._shared_embedding
        self.vector_store   = MoERecAgent._shared_vector_store
        self.id2name        = MoERecAgent._shared_id2name
        self.name2id        = MoERecAgent._shared_name2id
        self.id2rawid       = MoERecAgent._shared_id2rawid
        self.item_num       = MoERecAgent._shared_item_num
        self.device         = MoERecAgent._shared_device

        self._output_dir = (
            os.path.dirname(getattr(self.args, 'output_file', '') or '')
            or os.path.join(os.path.dirname(__file__), 'output')
        )

        self._build_pipeline()
        self._load_build_memory_template()

        # ── Phase 2: FeedbackScoreAdjuster (per-user, reset mỗi user) ────
        self.feedback_adjuster = FeedbackScoreAdjuster(
            negative_penalty  = getattr(args, 'fb_negative_penalty', 0.80),
            positive_boost    = getattr(args, 'fb_positive_boost',   1.20),
            max_penalty_rounds= getattr(args, 'fb_max_penalty',       3),
            llm_client        = self.llm,
            output_dir        = self._output_dir,
            dataset           = self.dataset,   # ← truyền dataset đã detect ở dòng 51-54
        )

    # ─────────────────────────────────────────────────────────────────────
    # Shared resource init (không thay đổi)
    # ─────────────────────────────────────────────────────────────────────

    @classmethod
    def _init_shared_resources(cls, args):
        import torch
        import pandas as pd
        from utils.model import SASRec

        if cls._shared_embedding is None:
            from langchain_huggingface import HuggingFaceEmbeddings
            embed_model = getattr(args, 'embed_model_name', "sentence-transformers/all-mpnet-base-v2")
            cls._shared_embedding = HuggingFaceEmbeddings(model_name=embed_model)

        if cls._shared_vector_store is None:
            faiss_path = getattr(args, 'faiss_db_path', None)
            if faiss_path and os.path.exists(faiss_path):
                from langchain_community.vectorstores import FAISS
                cls._shared_vector_store = FAISS.load_local(
                    folder_path=faiss_path,
                    embeddings=cls._shared_embedding,
                    allow_dangerous_deserialization=True,
                    distance_strategy="COSINE",
                )

        if cls._shared_gcn_embeddings is None:
            gcn_path = getattr(args, 'gcn_path', None)
            if gcn_path and os.path.exists(gcn_path):
                import torch as _t
                cls._shared_gcn_embeddings = _t.load(gcn_path, map_location='cpu', weights_only=True)

        if cls._shared_sasrec_model is None:
            import torch as _t
            device = _t.device("cuda" if _t.cuda.is_available() else "cpu")
            data_statis = pd.read_pickle(os.path.join(args.data_dir, 'data_statis.df'))
            seq_size, item_num = data_statis['seq_size'][0], data_statis['item_num'][0]
            ckpt = _t.load(args.model_path, map_location=device, weights_only=False)
            hidden_size = (
                ckpt.get('hidden_size', getattr(args, 'hidden_size', 64))
                if isinstance(ckpt, dict) else getattr(args, 'hidden_size', 64)
            )
            sasrec = SASRec(hidden_size, item_num, seq_size, 0.1, device).to(device)
            sasrec.load_state_dict(ckpt.get('model_state_dict', ckpt))
            sasrec.eval()

            id2name, name2id, id2rawid = {}, {}, {}
            with open(os.path.join(args.data_dir, 'id2name.txt'), encoding='utf-8') as f:
                for line in f:
                    ll = line.strip().split('::', 1)
                    if len(ll) == 2:
                        id2name[int(ll[0])] = ll[1].strip()
                        name2id[ll[1].strip()] = int(ll[0])
            rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
            if os.path.exists(rawid_path):
                with open(rawid_path, encoding='utf-8') as f:
                    for line in f:
                        ll = line.strip().split('::')
                        if len(ll) >= 2:
                            id2rawid[int(ll[0])] = ll[1].strip()

            cls._shared_sasrec_model  = sasrec
            cls._shared_id2name       = id2name
            cls._shared_name2id       = name2id
            cls._shared_id2rawid      = id2rawid
            cls._shared_seq_size      = seq_size
            cls._shared_item_num      = item_num
            cls._shared_device        = device

    # ─────────────────────────────────────────────────────────────────────
    # Pipeline build (không thay đổi logic, giữ Reranker object cho compat)
    # ─────────────────────────────────────────────────────────────────────

    def _build_pipeline(self):
        shared = {
            'model':             self.sasrec_model,
            'gcn_embeddings':    self.gcn_embeddings,
            'vector_store':      self.vector_store,
            'embedding_function':self.embedding_fn,
            'id2name':           self.id2name,
            'name2id':           self.name2id,
            'id2rawid':          self.id2rawid,
            'item_num':          self.item_num,
            'device':            self.device,
            'dataset':           self.dataset,
        }
        self.seq_scorer = SeqScorer.from_shared(shared)
        self.gcn_scorer = GCNScorer.from_shared(shared) if self.gcn_embeddings is not None else None
        self.sem_scorer = SemanticScorer.from_shared(shared)

        self.retriever = CandidateRetriever(
            seq_scorer   = self.seq_scorer,
            gcn_scorer   = self.gcn_scorer,
            sem_scorer   = self.sem_scorer,
            config       = self.cfg.retrieval,
            use_seq      = self.cfg.use_seq,
            use_gcn      = self.cfg.use_gcn and self.gcn_scorer is not None,
            use_semantic = self.cfg.use_semantic and self.vector_store is not None,
        )
        self.gating = GatingNetwork(
            cfg        = self.cfg.gating,
            model_path = getattr(self.args, 'gating_model_path', self.cfg.gating_model_path),
            device     = self.device,
        )
        gcn_norm_ref = self.gcn_scorer.gcn_norm if self.gcn_scorer is not None else None
        self.fuser   = MoEFusion(gating=self.gating, cfg=self.cfg, gcn_norm=gcn_norm_ref)

        # Giữ Reranker object để không break các import khác,
        # nhưng Phase 1: KHÔNG gọi reranker.rerank() trong act()
        self.reranker = Reranker.from_shared(
            shared     = shared,
            llm        = self.llm,
            mode       = getattr(self.args, 'reranker_mode', 'embed_only'),
            enabled    = self.cfg.use_reranker,
            top_llm    = 20,
            output_dir = self._output_dir,
        )
        self.combiner = ScoreCombiner(cfg=self.cfg.scoring)

    # ─────────────────────────────────────────────────────────────────────
    # Filtered reviews (không thay đổi)
    # ─────────────────────────────────────────────────────────────────────

    def _get_filtered_reviews(self, data: dict) -> list:
        tool = data.get('interaction_tool')
        if not tool:
            return []
        try:
            all_reviews = tool.get_reviews(user_id=str(data.get('id', '')))
            id2rawid = data.get('id2rawid', {})
            candidate_raw_ids = set()
            for inner_id in data.get('cans', []):
                raw_id = id2rawid.get(inner_id)
                if raw_id:
                    candidate_raw_ids.add(str(raw_id))
            return [r for r in all_reviews
                    if str(r.get('item_id', '')) not in candidate_raw_ids]
        except Exception:
            return []

    # ─────────────────────────────────────────────────────────────────────
    # act() — Phase 1: bypass Reranker | Phase 2: inject FeedbackAdjuster
    # ─────────────────────────────────────────────────────────────────────

    def act(
        self,
        data:           dict,
        reason:         str        = None,
        item:           str        = None,
        epoch:          int        = None,
        rejected_items: List[str]  = None,
    ) -> Tuple[str, List[str], dict]:

        raw_id  = data.get('id', data.get('user_id', 'unknown'))
        user_id = str(raw_id[0] if isinstance(raw_id, list) else raw_id)
        data['id'] = user_id

        if epoch is None:
            epoch = len(self.memory) + 1

        try:
            gt_item = data.get('correct_answer', '').strip()
            if 'reviews' not in data:
                reviews = self._get_filtered_reviews(data)
                if reviews:
                    data['reviews'] = reviews

            # ── Step 1: Retrieval ─────────────────────────────────────────
            union_names, signal_scores = self.retriever.retrieve_from_data(data)

            # ── Step 2: MoE Fusion ────────────────────────────────────────
            c_m, fused_scores, fusion_debug = self.fuser.fuse_from_data(
                data=data, signal_scores=signal_scores, debug=True
            )
            if not c_m:
                c_m = union_names[:self.cfg.retrieval.top_M]

            avg_gates  = fusion_debug.get('avg_gates', {'seq': 0.0, 'gcn': 0.0, 'sem': 0.0})
            moe_conf   = moe_confidence_score(fused_scores)

            # ── Step 3 (Phase 2): Feedback Score Adjustment ───────────────
            # Chỉ active từ epoch 2 trở đi (epoch 1 chưa có feedback)
            if epoch > 1:
                fused_scores = self.feedback_adjuster.adjust(fused_scores)
                # Re-sort c_m theo adjusted scores
                c_m = sorted(
                    fused_scores,
                    key=lambda x: fused_scores.get(x, 0.0),
                    reverse=True,
                )[:self.cfg.retrieval.top_M]
                print(
                    f"[FeedbackAdjuster] epoch={epoch} | "
                    f"rejected={self.feedback_adjuster.get_rejected_items()} | "
                    f"praised={self.feedback_adjuster.get_praised_items()}"
                )

            # ── Step 4 (Phase 1): Direct top-K — NO Reranker ─────────────
            c_k = c_m[:self.cfg.retrieval.top_K]

            # Build s0 rank scores cho debug
            s0_rank_scores = {
                item_name: 1.0 / math.log2(rank + 2)
                for rank, item_name in enumerate(c_m)
            }

            # [COMMENTED OUT Reranker for Feedback-only testing]
            # ranked, rerank_scores, explanation = self.reranker.rerank(data=data, c_m=c_m, id2name=self.id2name, name2id=self.name2id, memory=self.memory)

            # cm_fused = {name: fused_scores.get(name, 0.0) for name in c_m}
            # c_k_raw, s1_scores, combine_debug = self.combiner.combine_from_pipeline(fused_scores=cm_fused, rerank_scores=rerank_scores, data=data, args=self.args, top_k=self.cfg.retrieval.top_M, epoch=epoch)

            # c_k_sorted = sorted(c_k_raw, key=lambda x: s1_scores.get(x, 0.0), reverse=True)
            # c_k = c_k_sorted[:self.cfg.retrieval.top_K]



            explanation = (
                f"MoE recommendation (epoch={epoch}, "
                f"gates=seq:{avg_gates.get('seq',0):.2f}/"
                f"gcn:{avg_gates.get('gcn',0):.2f}/"
                f"sem:{avg_gates.get('sem',0):.2f})"
            )

            debug_info = {
                'gt_item':                  gt_item,
                'alpha':                    getattr(self.args, 'alpha', 0.5), # Fallback if combiner skipped
                'moe_confidence':           moe_conf,
                'top_M_size':               len(c_m),
                'avg_gates':                avg_gates,
                'c_m_top_k_before_rerank':  c_m[:self.cfg.retrieval.top_K],
                'c_k_final_after_rerank':   c_k,
                'feedback_adjuster_state':  self.feedback_adjuster.summary(),
                'scores_breakdown': {
                    item_name: {
                        's0_moe':      fused_scores.get(item_name, 0.0),
                        's_rerank':    0.0,
                        's1_final':    fused_scores.get(item_name, 0.0),
                    }
                    for item_name in c_k
                },
            }

            return explanation, c_k, debug_info

        except Exception as e:
            traceback.print_exc()
            return (
                "MoE pipeline failed.",
                data.get('cans_name', [])[:self.cfg.retrieval.top_K],
                {'error': str(e)},
            )

    # ─────────────────────────────────────────────────────────────────────
    # Memory (Phase 3: cập nhật FeedbackAdjuster trong update_memory)
    # ─────────────────────────────────────────────────────────────────────

    def build_memory(self, info: dict) -> str:
        rec_item_str = (
            ', '.join(info['rec_item_list'])
            if isinstance(info.get('rec_item_list'), list)
            else info.get('rec_item', '')
        )
        return self.rec_build_memory.format(
            info['epoch'],
            rec_item_str,
            info.get('rec_reason', ''),
            info.get('user_reason', ''),
        )

    def update_memory(self, info: dict):
        self.info_list.append(info)
        new_memory = self.build_memory(info)
        self.memory.append(new_memory)
        print(f"\n[MoERecAgent] New Memory built:\n{new_memory}\n")

        # ── Phase 3: Cập nhật FeedbackAdjuster sau mỗi round reject ──────
        self.feedback_adjuster.update_from_memory(
            user_reason    = info.get('user_reason', ''),
            rec_item_list  = info.get('rec_item_list', []),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Helpers (không thay đổi)
    # ─────────────────────────────────────────────────────────────────────

    def _load_build_memory_template(self):
        try:
            if 'amazon' in self.args.data_dir:
                from constant.amazon_prior_model_prompt import rec_build_memory
            elif 'goodreads' in self.args.data_dir:
                from constant.goodreads_prior_model_prompt import rec_build_memory
            elif 'yelp' in self.args.data_dir:
                from constant.yelp_prior_model_prompt import rec_build_memory
            else:
                rec_build_memory = "Round {}: Recommended {} because {}. Rejected because {}."
        except ImportError:
            rec_build_memory = "Round {}: Recommended {} because {}. Rejected because {}."
        self.rec_build_memory = rec_build_memory

    def get_shared_sasrec(self) -> dict:
        return {
            'model':    MoERecAgent._shared_sasrec_model,
            'id2name':  MoERecAgent._shared_id2name,
            'name2id':  MoERecAgent._shared_name2id,
            'id2rawid': MoERecAgent._shared_id2rawid,
            'seq_size': MoERecAgent._shared_seq_size,
            'item_num': MoERecAgent._shared_item_num,
            'device':   MoERecAgent._shared_device,
        }