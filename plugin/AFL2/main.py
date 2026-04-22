import argparse
import os
import time
import json
import pandas as pd
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import math
 
current_dir  = os.path.dirname(os.path.abspath(__file__))  # plugin/AFL2
parent_dir   = os.path.dirname(current_dir)                 # plugin
root_dir     = os.path.dirname(parent_dir)                  # AgentRecBench
 
sys.path.append(parent_dir)
sys.path.append(root_dir)
 
# --- Thêm baseline vào sys.path để tìm thấy websocietysimulator ---
baseline_dir = os.path.join(root_dir, 'baseline')
if baseline_dir not in sys.path:
    sys.path.insert(0, baseline_dir)
 
from utils.dialogue_manager import recommend, error_handler
from utils.data_processor import load_candidate_map, load_item_name_map, prepare_merge_data, build_candidate_order
from AFL2.utils.save_result import save_final_metrics
from utils.rw_process import append_jsonl
from dataset.general_dataset import GeneralDataset
from utils.agent import UserModelAgent, RecAgent
 
# --- Import InteractionTool ---
from websocietysimulator.tools import CacheInteractionTool
from utils.hybrid_ranking import tune_alpha
 
 
# ============================================================
# Argument Parser
# ============================================================
 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--input_json_file', type=str, default='')
    parser.add_argument('--candidate_dir', type=str, default=None)
    parser.add_argument('--item_mapping_file', type=str, default=None)
    parser.add_argument('--stage', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--cans_num', type=int, default=20)
    parser.add_argument('--max_samples', type=int, default=-1)
    parser.add_argument('--sep', type=str, default=', ')
    parser.add_argument('--max_epoch', type=int, default=3)
    parser.add_argument('--output_file', type=str, default='./output/dialogue_results.jsonl')
    parser.add_argument('--model', type=str, default='qwen-small')
    parser.add_argument("--base_url", type=str, default="http://localhost:8036/v1",
                        help="server vLLM")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--max_retry_num', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=333)
    parser.add_argument('--mp', type=int, default=4)
    parser.add_argument("--save_info", action="store_true")
    parser.add_argument("--save_rec_dir", type=str, default='./output/rec_logs')
    parser.add_argument("--save_user_dir", type=str, default='./output/user_logs')
    parser.add_argument('--hidden_size', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--result_file', type=str, default='evaluation_summary.json')
 
    # ARAG args
    parser.add_argument('--use_arag', action='store_true')
    parser.add_argument('--faiss_db_path', type=str, default=None)
    parser.add_argument('--gcn_path', type=str, default=None)
    parser.add_argument('--nli_threshold', type=float, default=5.5)
    parser.add_argument('--embed_model_name', type=str,
                        default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--use_hybrid', action='store_true',
                        help="Dùng hybrid SASRec+ARAG ranking")
 
    # --- Thêm arg cho raw data dir ---
    parser.add_argument('--raw_data_dir', type=str,
                        default=None,
                        help='Path to raw dataset dir chứa review.json, item.json, user.json')
 
    return parser.parse_args()
 
 
# ============================================================
# Shared Counters (thread/process-safe)
# ============================================================
 
def make_counters(manager):
    """
    Tạo shared counters an toàn với multiprocessing.Manager.
    Tất cả processes/threads đều đọc/ghi vào cùng một vùng nhớ
    được quản lý bởi Manager server process.
    """
    return {
        'finish_num':   manager.Value('i', 0),
        'correct_hit1': manager.Value('i', 0),
        'correct_hit3': manager.Value('i', 0),
        'correct_hit5': manager.Value('i', 0),
        'total_ndcg5':  manager.Value('d', 0.0),
        'total':        manager.Value('i', 0),
        'lock':         manager.Lock(),
        'hybrid_logs':  manager.list(),
    }
 
 
# ============================================================
# Safe Callback (dùng Lock để tránh race condition)
# ============================================================
 
def setcallback_safe(result, counters, args):
    """
    Callback thread/process-safe.
 
    - Ghi từng bước vào output file TRƯỚC khi lock (I/O không cần lock
      vì append_jsonl đã dùng 'a' mode, file-level lock của OS đủ an toàn
      cho các dòng JSONL độc lập).
    - Chỉ lock khi cập nhật shared counters để tránh race condition.
    """
    data_list, hit_at_n, _args, hybrid_log = result
 
    # Ghi output file (không cần lock, mỗi dòng JSONL là atomic)
    for step in data_list:
        append_jsonl(args.output_file, step)
 
    # Cập nhật counters với lock
    with counters['lock']:
        if hybrid_log:
            counters['hybrid_logs'].extend(hybrid_log)
 
        counters['finish_num'].value += 1
 
        if hit_at_n.get(1):
            counters['correct_hit1'].value += 1
        if hit_at_n.get(3):
            counters['correct_hit3'].value += 1
        if hit_at_n.get(5):
            counters['correct_hit5'].value += 1
 
        rank = hit_at_n.get('rank')
        if rank is not None and rank <= 5:
            counters['total_ndcg5'].value += 1.0 / math.log2(rank + 1)
 
        # Snapshot để in ra ngoài lock
        fn   = counters['finish_num'].value
        tot  = counters['total'].value
        h1   = counters['correct_hit1'].value
        h3   = counters['correct_hit3'].value
        h5   = counters['correct_hit5'].value
        ndcg = counters['total_ndcg5'].value
 
    agent_tag = "ARAG" if getattr(args, 'use_arag', False) else "AFL"
    print(
        f"[{agent_tag}][{fn}/{tot}] "
        f"Hit@1: {h1/fn*100:.2f}% | "
        f"Hit@3: {h3/fn*100:.2f}% | "
        f"Hit@5: {h5/fn*100:.2f}% | "
        f"NDCG@5: {ndcg/fn:.4f}",
        flush=True,
    )
 
 
# ============================================================
# Main
# ============================================================
 
def main(args):
    # --- Banner ---
    if args.use_arag:
        print("=" * 60)
        print("  MODE: AFL + ARAG (Agentic RAG) Integration")
        print(f"  FAISS DB:      {args.faiss_db_path}")
        print(f"  GCN Path:      {args.gcn_path}")
        print(f"  NLI Threshold: {args.nli_threshold}")
        print(f"  Raw Data Dir:  {args.raw_data_dir}")
        print(f"  Workers:       {args.mp} (ThreadPool, fork)")
        print("=" * 60)
    else:
        print("=" * 60)
        print(f"  MODE: Vanilla AFL (no ARAG), mp={args.mp}")
        print("=" * 60)
 
    # --- 1. Load dataset ---
    dataset  = GeneralDataset(args, stage=args.stage)
    data_map = {str(d['id']): d for d in dataset}
 
    # --- 2. Load input list ---
    with open(args.input_json_file, 'r', encoding='utf-8') as f:
        new_input_list = json.load(f)
 
    # --- 3. Load candidate & item maps ---
    print(args.candidate_dir)
    candidate_map = load_candidate_map(args.candidate_dir)
    item_name_map = load_item_name_map(args.item_mapping_file)
 
    # --- 4. SASRec prior recommendation ---
    temp_args = argparse.Namespace(**vars(args))
    temp_args.model = 'sasrec_inference'
    sasrec_tool = UserModelAgent(temp_args, mode='prior_rec')
 
    merge_data_list, skipped = prepare_merge_data(
        new_input_list, data_map, candidate_map, item_name_map, sasrec_tool, args
    )
 
    # --- 5. Sort by candidate order ---
    candidate_order = build_candidate_order(args.candidate_dir)
    if candidate_order:
        merge_data_list.sort(
            key=lambda d: candidate_order.get(str(d['id']), float('inf'))
        )
        print(f"[Main] Sorted merge_data_list by candidate file order "
              f"({len(candidate_order)} entries)")
 
    if args.max_samples > 0:
        merge_data_list = merge_data_list[:]
 
    # --- 6. ARAG setup: InteractionTool + id2rawid ---
    if args.use_arag:
        if not args.raw_data_dir:
            args.raw_data_dir = os.path.join(root_dir, 'dataset', 'output_data_all')
        args.raw_data_dir = os.path.abspath(args.raw_data_dir)
 
        if not os.path.exists(args.raw_data_dir):
            raise FileNotFoundError(
                f"[Main] raw_data_dir không tồn tại: {args.raw_data_dir}\n"
                f"Hãy truyền --raw_data_dir=<đường dẫn tuyệt đối đến thư mục chứa item.json>"
            )
 
        print(f"[Main] Initializing InteractionTool from {args.raw_data_dir} ...")
        interaction_tool = CacheInteractionTool(data_dir=args.raw_data_dir)
        print("[Main] InteractionTool ready.")
 
        id2rawid = {}
        rawid_path = os.path.join(args.data_dir, 'id2rawid.txt')
        if os.path.exists(rawid_path):
            with open(rawid_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('::')
                    if len(parts) >= 2:
                        id2rawid[int(parts[0])] = parts[1].strip()
            print(f"[Main] Loaded id2rawid.txt ({len(id2rawid)} entries).")
        else:
            print(f"[Main] WARNING: id2rawid.txt not found at {rawid_path}.")
 
        for d in merge_data_list:
            d['interaction_tool'] = interaction_tool
            d['id2rawid']         = id2rawid
            d['task_set']         = args.data_dir
 
    # --- 7. Run với Manager để shared counters an toàn ---
    total = len(merge_data_list)
    print(f"Ready: {total} samples. Skipped: {skipped}")
 
    with multiprocessing.Manager() as manager:
        counters = make_counters(manager)
        counters['total'].value = total
 
        if args.use_arag:
            # ----------------------------------------------------------------
            # ARAG mode: dùng ThreadPoolExecutor
            # Lý do: LangGraph dùng asyncio/event-loop bên trong, không tương
            # thích với os.fork() của multiprocessing trên Linux/macOS.
            # ThreadPool không fork → LangGraph hoạt động bình thường.
            # I/O-bound tasks (LLM API calls) hưởng lợi từ threading.
            # ----------------------------------------------------------------
            effective_workers = max(1, args.mp)
            print(f"[ARAG] ThreadPoolExecutor, workers={effective_workers}")
 
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {
                    executor.submit(recommend, data, args): data
                    for data in merge_data_list
                }
                for future in tqdm(as_completed(futures), total=total, desc="Processing (ARAG-Thread)"):
                    try:
                        result = future.result()
                        setcallback_safe(result, counters, args)
                    except Exception as e:
                        error_handler(e)
 
        elif args.mp <= 1:
            # ----------------------------------------------------------------
            # Single-process mode: debug / low-memory
            # ----------------------------------------------------------------
            print("[AFL] Single-process mode")
            for data in tqdm(merge_data_list, desc="Processing (Single)"):
                try:
                    result = recommend(data, args)
                    setcallback_safe(result, counters, args)
                except Exception as e:
                    error_handler(e)
 
        else:
            # ----------------------------------------------------------------
            # AFL Multiprocessing mode: Pool.apply_async
            # Callback chạy trên main process → an toàn với Manager counters.
            # Không dùng lambda để tránh pickle error trên một số OS.
            # ----------------------------------------------------------------
            print(f"[AFL] Multiprocessing Pool, workers={args.mp}")
 
            # Wrapper để truyền counters và args vào callback
            # (pool callback chỉ nhận 1 argument là result)
            def _callback(result):
                setcallback_safe(result, counters, args)
 
            pool = multiprocessing.Pool(processes=args.mp)
            for data in merge_data_list:
                pool.apply_async(
                    recommend,
                    args=(data, args),
                    callback=_callback,
                    error_callback=error_handler,
                )
            pool.close()
            pool.join()
 
        # --- 8. Đọc final values SAU khi tất cả workers xong ---
        final_hit1 = counters['correct_hit1'].value
        final_hit3 = counters['correct_hit3'].value
        final_hit5 = counters['correct_hit5'].value
        final_ndcg = counters['total_ndcg5'].value
        final_logs = list(counters['hybrid_logs'])
 
    # --- 9. Lưu kết quả ---
    save_final_metrics(args, total, final_hit1, final_hit3, final_hit5, final_ndcg)
 
    if getattr(args, 'use_hybrid', False) and final_logs:
        print("\n" + "=" * 30)
        print("HYBRID RANKING ANALYSIS")
        print("=" * 30)
 
        tuning_results = tune_alpha(final_logs)
 
        tuning_file = args.output_file.replace('.jsonl', '_hybrid_tuning.json')
        with open(tuning_file, 'w') as f:
            json.dump(tuning_results, f, indent=4)
        print(f"Detailed hybrid analysis saved to: {tuning_file}")
 
 
if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
 
    args = get_args()
    random.seed(args.seed)
    main(args)