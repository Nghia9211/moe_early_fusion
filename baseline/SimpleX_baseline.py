"""
SimpleX Standalone Baseline
============================
Evaluates a pre-trained SimpleX model through the WebSocietySimulator
framework, following the same pattern as DummyAgent_baseline.py.

Workflow:
  1. Load pre-trained user/item embeddings from plugin/simplex/embeddings/{dataset}/
  2. For each task, rank the candidate_list by cosine similarity to the user embedding
  3. Evaluate via Simulator → HR@1, HR@3, HR@5, NDCG@5

Usage:
    cd baseline/
    python SimpleX_baseline.py --task_set yelp --scenario classic
    python SimpleX_baseline.py --task_set amazon --scenario classic
    python SimpleX_baseline.py --task_set amazon --scenario user_cold_start
    python SimpleX_baseline.py --task_set goodreads --scenario classic
"""

import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

from websocietysimulator import Simulator
from websocietysimulator.agent import RecommendationAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SimpleX Recommendation Agent
# ---------------------------------------------------------------------------

class SimpleXRecommendationAgent(RecommendationAgent):
    """
    Pure CF agent using pre-trained SimpleX embeddings.
    No LLM required.  Ranks candidates by cosine similarity.
    """

    # Class-level embedding cache (shared across all instances)
    _user_embs: dict = {}   # { user_id_str: tensor }
    _item_embs: dict = {}   # { item_id_str: tensor }
    _dataset: str = ""

    @classmethod
    def load_embeddings(cls, emb_dir: str, dataset: str):
        """
        Load user and item embeddings from disk.
        Called once before Simulator.run_simulation().

        Args:
            emb_dir : path to the root embeddings directory
                      (e.g. '../plugin/simplex/embeddings')
            dataset : 'amazon' | 'yelp' | 'goodreads'
        """
        dataset_dir = os.path.join(emb_dir, dataset)

        user_path = os.path.join(dataset_dir, "user_embs.pt")
        item_path = os.path.join(dataset_dir, "item_embs.pt")

        if not os.path.exists(user_path) or not os.path.exists(item_path):
            raise FileNotFoundError(
                f"Embeddings not found in '{dataset_dir}'.\n"
                f"Run 'cd ../plugin/simplex && python train.py --dataset {dataset}' first."
            )

        logger.info(f"Loading SimpleX embeddings from {dataset_dir} …")
        cls._user_embs = torch.load(user_path, map_location="cpu")
        cls._item_embs = torch.load(item_path, map_location="cpu")
        cls._dataset   = dataset
        logger.info(
            f"  Users: {len(cls._user_embs):,}  |  Items: {len(cls._item_embs):,}"
        )

    def __init__(self, llm=None):
        """llm is accepted but ignored — SimpleX doesn't need an LLM."""
        super().__init__(llm=llm)

    def workflow(self):
        """
        Rank the candidate_list by cosine similarity to the user embedding.

        Returns:
            List of item_id strings, ranked most → least relevant.
            Falls back to the original candidate_list order if the user is
            unknown (cold-start scenario).
        """
        user_id    = self.task["user_id"]
        candidates = self.task["candidate_list"]  # list of item_id strings

        # ---- Look up user embedding ----------------------------------------
        u_emb = self._user_embs.get(user_id)
        if u_emb is None:
            # Unknown user (user cold-start) → return original order
            logger.debug(f"Unknown user '{user_id}', returning original order.")
            return candidates

        u_emb = F.normalize(u_emb.unsqueeze(0), dim=-1)  # (1, d)

        # ---- Collect item embeddings for candidates ------------------------
        scores = []
        for item_id in candidates:
            i_emb = self._item_embs.get(item_id)
            if i_emb is None:
                # Unknown item → score of -1 (will be ranked last)
                scores.append(-1.0)
            else:
                i_emb = F.normalize(i_emb.unsqueeze(0), dim=-1)  # (1, d)
                score = (u_emb * i_emb).sum().item()
                scores.append(score)

        # ---- Sort candidates by score descending ---------------------------
        ranked = [item_id for _, item_id in
                  sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

        return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run SimpleX Baseline via WebSocietySimulator")
    parser.add_argument(
        "--task_set", default="yelp",
        choices=["amazon", "yelp", "goodreads"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--scenario", default="classic",
        choices=["classic", "user_cold_start", "item_cold_start"],
        help="Evaluation scenario"
    )
    parser.add_argument(
        "--data_dir", default="../dataset/output_data_all",
        help="Path to output_data_all directory"
    )
    parser.add_argument(
        "--emb_dir", default="../plugin/simplex/embeddings",
        help="Path to SimpleX embeddings root directory"
    )
    parser.add_argument(
        "--tasks_dir", default="../dataset/tasks5",
        help="Path to tasks directory (contains classic/, user_cold_start/, …)"
    )
    parser.add_argument(
        "--num_tasks", type=int, default=500,
        help="Number of tasks to evaluate (default: 500)"
    )
    parser.add_argument(
        "--max_workers", type=int, default=20,
        help="Number of parallel threads"
    )
    args = parser.parse_args()

    task_set = args.task_set
    scenario = args.scenario

    # ---- Load embeddings ---------------------------------------------------
    SimpleXRecommendationAgent.load_embeddings(args.emb_dir, task_set)

    # ---- Setup Simulator ---------------------------------------------------
    simulator = Simulator(
        data_dir=args.data_dir,
        device="cpu",  # SimpleX inference is CPU-based (embedding lookup)
        cache=True,
    )
    simulator.set_task_and_groundtruth(
        task_dir=os.path.join(args.tasks_dir, scenario, task_set, "tasks"),
        groundtruth_dir=os.path.join(args.tasks_dir, scenario, task_set, "groundtruth"),
    )
    simulator.set_agent(SimpleXRecommendationAgent)
    # No LLM needed; set_llm can be skipped but Simulator may expect it
    # → pass None explicitly to avoid AttributeError in some simulator versions
    simulator.set_llm(None)

    # ---- Run ---------------------------------------------------------------
    logger.info(f"Running SimpleX baseline on {task_set} / {scenario} …")
    simulator.run_simulation(
        number_of_tasks=args.num_tasks,
        enable_threading=True,
        max_workers=args.max_workers,
    )
    evaluation_results = simulator.evaluate()

    # ---- Save results ------------------------------------------------------
    out_dir = os.path.join("results", scenario)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"evaluation_results_SimpleX_{task_set}.json")

    with open(out_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    logger.info(f"Results saved to {out_path}")
    print(f"\n=== SimpleX Evaluation Results ({task_set} / {scenario}) ===")
    print(json.dumps(evaluation_results, indent=4))


if __name__ == "__main__":
    main()
