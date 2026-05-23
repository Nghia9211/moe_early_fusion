import os
import json
from tqdm import tqdm


def build_candidate_order(candidate_dir):
    """
    Tạo mapping: user_id → thứ tự (index) theo số thứ tự file trong candidate_dir.
    task_0.json → 0, task_1.json → 1, ..., task_99.json → 99
    
    Dùng để sort merge_data_list cho khớp thứ tự candidate dir.
    """
    order_map = {}
    if not candidate_dir or not os.path.exists(candidate_dir):
        return order_map

    candidate_files = [f for f in os.listdir(candidate_dir)
                       if f.startswith('task_') and f.endswith('.json')]
    # Sort theo số trong tên file: task_0, task_1, ..., task_100
    candidate_files.sort(key=lambda f: int(f.replace('task_', '').replace('.json', '')))

    for idx, file_name in enumerate(candidate_files):
        try:
            with open(os.path.join(candidate_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                uid = str(entry.get('user_id'))
                order_map[uid] = idx
        except:
            continue

    print(f"[CandidateOrder] Built order map for {len(order_map)} users "
          f"(task_0 → task_{len(order_map)-1})")
    return order_map


def load_candidate_map(candidate_dir):
    candidate_map = {}
    if not candidate_dir or not os.path.exists(candidate_dir):
        return candidate_map
    
    candidate_files = [f for f in os.listdir(candidate_dir) if f.startswith('task_') and f.endswith('.json')]
    for file_name in tqdm(candidate_files, desc="Loading Candidates"):
        try:
            with open(os.path.join(candidate_dir, file_name), 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry = data[0] if isinstance(data, list) else data
                candidate_map[str(entry.get('user_id'))] = entry.get('candidate_list', [])
        except: continue
    return candidate_map

def load_item_name_map(mapping_file):
    name_map = {}
    if not mapping_file or not os.path.exists(mapping_file):
        return name_map
    with open(mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                name = item.get('title') or item.get('name') or "Unknown"
                asin = item.get('asin') or item.get('item_id') or "Unknown"
                unique_name = f"{name.strip()} [{asin.strip()}]"
                name_map[str(item.get('item_id')).strip()] = unique_name
            except: continue
    return name_map
def prepare_merge_data(new_input_list, data_map, candidate_map, item_id_to_name_map, user_agent_sasrec, args):
    merge_data_list = []
    skipped = 0
    padding_id = 0

    # Build reverse mapping: raw_id (ASIN) → inner_id
    # id2rawid maps inner_id → ASIN, we need ASIN → inner_id
    rawid2id = {}
    if hasattr(user_agent_sasrec, 'id2rawid') and user_agent_sasrec.id2rawid:
        rawid2id = {v: k for k, v in user_agent_sasrec.id2rawid.items()}
    
    total_missing = 0
    total_invalid = 0
    missing_examples = []  # Lưu vài ví dụ để debug

    for entry in tqdm(new_input_list, desc="Merging Data"):
        user_id = str(entry.get('user_id'))
        gt_item_id = str(entry.get('item_id', '')).strip()
        
        # --- DEBUG GROUND TRUTH ---
        # Thử tìm inner_id từ ASIN trước
        gt_inner_id = rawid2id.get(gt_item_id)
        name_from_map = item_id_to_name_map.get(gt_item_id)
        name_from_model = user_agent_sasrec.id2name.get(gt_inner_id) if gt_inner_id is not None else \
                          user_agent_sasrec.id2name.get(int(gt_item_id) if gt_item_id.isdigit() else -1)
        
        if not name_from_map and not name_from_model:
            print(f"\n[CRITICAL] Ground Truth Item {gt_item_id} (User: {user_id}) KHÔNG TỒN TẠI trong item.json lẫn id2name.txt!")
        
        gt_item_name = name_from_map or name_from_model or gt_item_id

        if user_id not in data_map:
            data = {'id': user_id, 'uid': user_id, 'seq': [padding_id] * user_agent_sasrec.seq_size, 'seq_str': "Empty History", 'len_seq': 0, 'seq_unpad': []}
        else:
            data = data_map[user_id].copy()

        data['correct_answer'] = gt_item_name
        
        if user_id in candidate_map:
            new_ids, new_names = [], []
            for rid in candidate_map[user_id]:
                rid_str = str(rid).strip()
                
                # Bước 1: Thử tra trực tiếp ASIN → inner_id qua rawid2id
                iid = rawid2id.get(rid_str)
                
                if iid is not None:
                    # Tìm thấy inner_id → lấy tên từ id2name
                    name = user_agent_sasrec.id2name.get(iid, rid_str)
                else:
                    # Bước 2: Fallback - thử tìm tên rồi tra name2id (cho trường hợp rid là inner_id số)
                    name = item_id_to_name_map.get(rid_str) or user_agent_sasrec.id2name.get(int(rid_str) if rid_str.isdigit() else -1) or rid_str
                    iid = user_agent_sasrec.name2id.get(name)
                
                # Item không tồn tại trong model SASRec (cold-start item)
                if iid is None:
                    total_missing += 1
                    if len(missing_examples) < 5:
                        missing_examples.append(f"  '{name}' (ID: {rid_str})")
                    continue
                
                if iid >= user_agent_sasrec.item_num:
                    total_invalid += 1
                    continue

                new_names.append(name)
                new_ids.append(iid)
            
            # Kiểm tra nếu sau khi lọc không còn ứng viên nào
            if len(new_ids) == 0:
                print(f"[WARNING] User {user_id} không có ứng viên nào hợp lệ sau khi lọc!")
                skipped += 1
                continue

            data['cans'] = new_ids
            data['cans_str'] = args.sep.join(new_names)
            data['len_cans'] = len(new_ids)
            data['cans_name'] = new_names
        else:
            skipped += 1
            continue

        try:
            data['prior_answer'] = user_agent_sasrec.model_generate(data['seq'], data['len_seq'], data['cans'])
            merge_data_list.append(data)
        except Exception as e:
            # In chi tiết lỗi tại đây
            print(f"\n[ERROR] Lỗi tại model_generate cho User {user_id}: {e}")
            print(f"Dữ liệu gây lỗi - Cans: {data.get('cans')}, Item_num: {user_agent_sasrec.item_num}")
            skipped += 1
    
    # In tóm tắt cuối cùng thay vì spam từng item
    if total_missing > 0:
        print(f"\n[MISSING ITEMS SUMMARY] {total_missing} candidate items không có trong model SASRec (cold-start items bị lọc khi tiền xử lý).")
        print(f"  → Nguyên nhân: Candidate list chứa ASIN không nằm trong id2rawid.txt/id2name.txt")
        print(f"  → Các item này đã bị bỏ qua (không ảnh hưởng kết quả).")
        if missing_examples:
            print(f"  → Ví dụ ({len(missing_examples)} mẫu):")
            for ex in missing_examples:
                print(ex)
    if total_invalid > 0:
        print(f"[INVALID IDs SUMMARY] {total_invalid} items có inner_id >= item_num, đã bị loại bỏ.")
            
    return merge_data_list, skipped