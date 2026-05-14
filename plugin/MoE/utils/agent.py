import os
import torch
import sys
import pandas as pd
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import model

sys.modules['SASRecModules_ori'] = model

from utils.helper_function import write_jsonl, read_jsonl, api_request
from utils.model import SASRec

class UserModelAgent:
    def __init__(self, args, mode='prior_rec', shared_sasrec=None):
        """
        Args:
            shared_sasrec: dict với các key 'model', 'id2name', 'name2id',
                           'id2rawid', 'seq_size', 'item_num', 'device'.
                           Nếu được truyền vào, bỏ qua load_model() và
                           build_id2name() để tránh load SASRec lần 2.
        """
        self.memory = []
        self.info_list = []
        self.args = args
        self.mode = mode
        self.load_prompt()

        if shared_sasrec is not None:
            # Tái sử dụng SASRec đã load từ ARAGRecAgent
            self.model    = shared_sasrec['model']
            self.id2name  = shared_sasrec['id2name']
            self.name2id  = shared_sasrec['name2id']
            self.id2rawid = shared_sasrec['id2rawid']
            self.seq_size = shared_sasrec['seq_size']
            self.item_num = shared_sasrec['item_num']
            self.device   = shared_sasrec['device']
            print("[UserModelAgent] Reusing shared SASRec — skipping reload.")
        else:
            # Load bình thường khi dùng độc lập (không có ARAG)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.id2name  = dict()
            self.name2id  = dict()
            self.id2rawid = dict()
            self.build_id2name()
            self.load_model()

    def build_id2name(self):
        if any(x in self.args.data_dir for x in ['yelp', 'amazon', 'goodreads']):
            item_path = os.path.join(self.args.data_dir, 'id2name.txt')
            with open(item_path, 'r', encoding='utf-8') as f:
                for l in f.readlines():
                    ll = l.strip('\n').split('::', 1)
                    self.id2name[int(ll[0])] = ll[1].strip()
                    self.name2id[ll[1].strip()] = int(ll[0])

            rawid_path = os.path.join(self.args.data_dir, 'id2rawid.txt')
            if os.path.exists(rawid_path):
                with open(rawid_path, 'r', encoding='utf-8') as f:
                    for l in f.readlines():
                        ll = l.strip('\n').split('::')
                        if len(ll) >= 2:
                            self.id2rawid[int(ll[0])] = ll[1].strip()
            else:
                print(f"[UserModelAgent] WARNING: id2rawid.txt not found at {rawid_path}. "
                      f"Run process_data.py to generate it.")
        else:
            raise ValueError("Invalid data dir: {}".format(self.args.data_dir))

    def load_model(self):
        print(f"Loading model from {self.args.model_path}")
        data_directory = self.args.data_dir
        data_statis = pd.read_pickle(os.path.join(data_directory, 'data_statis.df'))
        self.seq_size = data_statis['seq_size'][0]
        self.item_num = data_statis['item_num'][0]

        checkpoint = torch.load(self.args.model_path, map_location=self.device, weights_only=False)
        hidden_size = checkpoint.get('hidden_size', getattr(self.args, 'hidden_size', 64)) if isinstance(checkpoint, dict) else getattr(self.args, 'hidden_size', 64)
        self.model = SASRec(hidden_size, self.item_num, self.seq_size, 0.1, self.device)
        self.model.to(self.device)

        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            self.model = checkpoint

        self.model.eval()
        print("Load model weights success!")

    def load_prompt(self):
        if self.mode == 'prior_rec':
            if 'amazon' in self.args.data_dir:
                from constant.amazon_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'goodreads' in self.args.data_dir:
                from constant.goodreads_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            elif 'yelp' in self.args.data_dir:
                from constant.yelp_prior_model_prompt import user_system_prompt, user_user_prompt, user_memory_system_prompt, user_memory_user_prompt, user_build_memory, user_build_memory_2
            else:
                raise ValueError("Invalid dataset: {}".format(self.args.data_dir))
            self.user_system_prompt = user_system_prompt
            self.user_user_prompt = user_user_prompt
            self.user_memory_system_prompt = user_memory_system_prompt
            self.user_memory_user_prompt = user_memory_user_prompt
            self.user_build_memory = user_build_memory
            self.user_build_memory_2 = user_build_memory_2


    def act(self, data, reason=None, item_list=None, docstore_cache=None):
        if self.mode == 'prior_rec':

            def build_user_query(data):
                # Fields to strip from review history (noise / irrelevant metadata)
                _STRIP_FIELDS = {'date_added', 'date_updated', 'source', 'type',
                                 'timestamp', 'image', 'sub_item_id', 'date',
                                 'review_id', 'user_id'}

                parts = []
                reviews = data.get('reviews', [])
                if reviews:
                    if isinstance(reviews, list):
                        id2rawid         = data.get('id2rawid', self.id2rawid)
                        interaction_tool = data.get('interaction_tool')

                        # Build mapping from raw_id -> item_name
                        rawid2name = {str(raw): self.id2name.get(inner)
                                      for inner, raw in id2rawid.items() if inner in self.id2name}

                        candidate_ids = set()
                        for inner_id in data.get('cans', []):
                            raw_id = id2rawid.get(inner_id)
                            if raw_id:
                                candidate_ids.add(str(raw_id))

                        filtered_reviews = []
                        for r in reviews:
                            item_id_str = str(r.get('item_id', ''))
                            if item_id_str not in candidate_ids:
                                # 1. Bỏ các trường noise
                                r_copy = {k: v for k, v in r.items() if k not in _STRIP_FIELDS}

                                # 2. Thêm item_name
                                item_name = rawid2name.get(item_id_str)
                                if item_name:
                                    r_copy['item_name'] = item_name

                                # 3. Inject categories từ interaction_tool (nếu có)
                                if interaction_tool:
                                    try:
                                        fetched = interaction_tool.get_item(item_id=item_id_str)
                                        if fetched:
                                            cats = fetched.get('categories')
                                            if cats:
                                                if isinstance(cats, list):
                                                    cats = ', '.join(str(c) for c in cats)
                                                r_copy['categories'] = cats
                                    except Exception:
                                        pass

                                # 4. Đưa item_id và item_name lên đầu dict
                                ordered = {}
                                if 'item_id'    in r_copy: ordered['item_id']    = r_copy.pop('item_id')
                                if 'item_name'  in r_copy: ordered['item_name']  = r_copy.pop('item_name')
                                if 'categories' in r_copy: ordered['categories'] = r_copy.pop('categories')
                                ordered.update(r_copy)  # còn lại: stars, text, ...

                                filtered_reviews.append(ordered)
                        history_review = str(filtered_reviews)
                    else:
                        history_review = str(reviews)
                    try:
                        import tiktoken
                        enc = tiktoken.get_encoding("cl100k_base")
                        encoded = enc.encode(history_review)
                        if len(encoded) > 8000: history_review = enc.decode(encoded[:8000])
                    except Exception:
                        if len(history_review) > 6000: history_review = history_review[:6000]
                    parts.append(f"\n{history_review}")
                else:
                    seq_str = data.get('seq_str', '') or ''
                    if seq_str and seq_str.strip() and seq_str != 'Empty History':
                        words = seq_str.split()
                        parts.append(f"{' '.join(words[-80:])}")
                return "\n\n".join(parts) if parts else ""

            def enrich_candidates(items_input):
                if not items_input: return "None"
                items = [i.strip() for i in items_input.split(',')] if isinstance(items_input, str) else items_input
                interaction_tool = data.get('interaction_tool')
                id2rawid = data.get('id2rawid', self.id2rawid)
                if not interaction_tool:
                    return ", ".join(items) if isinstance(items, list) else items_input

                enriched = []
                keys = ['average_rating', 'stars', 'review_count', 'categories', 'description']
                for idx, item in enumerate(items, 1):
                    inner_id = self.name2id.get(item)
                    raw_id = id2rawid.get(inner_id) if inner_id is not None else None
                    if raw_id:
                        try:
                            fetched = interaction_tool.get_item(item_id=raw_id)
                            if fetched:
                                details = []
                                for k in keys:
                                    v = fetched.get(k)
                                    if v:
                                        if k == 'description' and isinstance(v, str) and len(v) > 150:
                                            v = v[:150] + '...'
                                        if k == 'categories' and isinstance(v, str) and len(v) > 100:
                                            v = v[:100] + '...'
                                        details.append(f"{k}: {v}")
                                if details:
                                    enriched.append(f'#{idx}: "{item}" — {", ".join(details)}')
                                else:
                                    enriched.append(f'#{idx}: "{item}"')
                                continue
                        except Exception: pass
                    enriched.append(f'#{idx}: "{item}"')
                return "\n".join(enriched)

            # Build user query based on reviews to avoid data leakage
            enriched_seq_str = build_user_query(data)

            # ── PATCH: Enrich Top-5 item_list với metadata từ interaction_tool ──
            if item_list is not None:
                items_raw = item_list if isinstance(item_list, list) \
                            else [i.strip() for i in str(item_list).split(',')]
                enriched_item_list = enrich_candidates(items_raw)
            else:
                enriched_item_list = "None"
            # ────────────────────────────────────────────────────────────────

            if len(self.memory) == 0:
                system_prompt = self.user_system_prompt.format(enriched_seq_str)
                # Avoid duplicating large history in user prompt if already in system prompt
                user_history_placeholder = "As detailed in your reading history above" 
                user_prompt = self.user_user_prompt.format(
                    user_history_placeholder,
                    enriched_item_list,
                    reason,
                )
            else:
                system_prompt = self.user_memory_system_prompt.format(enriched_seq_str)
                # Truncate memory to last 2 rounds to save context
                memory_str = '\n'.join(self.memory[-2:])
                user_history_placeholder = "As detailed in your reading history above"
                user_prompt = self.user_memory_user_prompt.format(
                    user_history_placeholder,
                    memory_str,
                    enriched_item_list,
                    reason,
                )
            # Call LLM first so we can log both prompt and response together
            response = api_request(system_prompt, user_prompt, self.args)
            try:
                out_dir = os.path.dirname(getattr(self.args, 'output_file', ''))
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                    log_path = os.path.join(out_dir, "user_agent_prompts.txt")
                    round_num   = len(self.memory) + 1
                    round_label = "INITIAL ROUND" if len(self.memory) == 0 else f"FEEDBACK ROUND (memory={len(self.memory)})"
                    SEP  = "=" * 80
                    SEP2 = "-" * 80
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n{SEP}\n")
                        f.write(f"[USER AGENT] User ID: {data.get('id', 'Unknown')} | ROUND {round_num} | {round_label}\n")
                        f.write(f"{SEP}\n")
                        f.write(f"[SYSTEM PROMPT]\n{SEP2}\n{system_prompt}\n{SEP2}\n")
                        f.write(f"[USER PROMPT]\n{SEP2}\n{user_prompt}\n{SEP2}\n")
                        f.write(f"[USER AGENT RESPONSE]\n{SEP2}\n{response}\n{SEP2}\n")
                        f.write(f"{SEP}\n")
            except Exception as e:
                print(f"Error logging prompt: {e}")
            return response
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))

    def pred_model(self, data, score):
        if len(self.memory) == 0:
            system_prompt = self.user_system_prompt.format(data['seq_str'])
            user_prompt = self.user_user_prompt.format(data['pred_item'], score)
        else:
            system_prompt = self.user_memory_system_prompt.format(data['seq_str'], data['cans_str'], '\n'.join(self.memory))
            user_prompt = self.user_memory_user_prompt.format(data['pred_item'], score)

        response = api_request(system_prompt, user_prompt, self.args)
        return response

    def build_memory(self, info):
        if info['user_reason'] is not None:
                return self.user_build_memory.format(info['epoch'], info['rec_item_list'], info['rec_reason'], info['user_reason'])
        else:
            return self.user_build_memory_2.format(info['epoch'], info['rec_item_list'], info['rec_reason'])

    def update_memory(self, info):
        self.info_list.append(info)
        new_memory = self.build_memory(info)
        self.memory.append(new_memory)
        print(f"\n[UserModelAgent] New Memory built:\n{new_memory}\n")

    def save_memory(self, path):
        write_jsonl(path, self.info_list)

    def load_memory(self, path):
        self.info_list = read_jsonl(path)
        self.memory = [self.build_memory(info) for info in self.info_list]

    def regenerate_prior(self, data):
        data['prior_answer'] = self.model_generate(
            data['seq'], data['len_seq'], data['cans']
        )

    def model_generate(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        topK = topK.numpy()[0]
        name_list = [self.id2name[id] for id in topK]
        len_ret = int(len(name_list) / 4)
        return ', '.join(name_list[:len_ret])

    def score(self, seq, len_seq, candidates):
        seq_b = [seq]
        len_seq_b = [len_seq]
        states = np.array(seq_b)
        states = torch.LongTensor(states).to(self.device)
        prediction = self.model.forward_eval(states, np.array(len_seq_b))

        sampling_idx = [True] * self.item_num
        cans_num = len(candidates)
        for i in candidates:
            sampling_idx.__setitem__(i, False)
        sampling_idxs = torch.stack([torch.tensor(sampling_idx)], dim=0)
        prediction = prediction.cpu().detach().masked_fill(sampling_idxs, prediction.min().item() - 1)
        values, topK = prediction.topk(cans_num, dim=1, largest=True, sorted=True)
        values = values.numpy()[0]
        topK = topK.numpy()[0]
        score_dict = {}
        for i in range(len(topK)):
            id = topK[i]
            score = values[i]
            name = self.id2name[id]
            score_dict[name] = score
        return score_dict