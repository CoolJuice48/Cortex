"""
Rebuild graph from existing TSBs
"""
import pickle
from pathlib import Path
from read_and_embed import TSB, build_graph, save_graph_and_tsbs

OUTPUT_DIR = Path('/Volumes/256GB FLASH/Cortex/backend/embed/output')

print("Loading TSBs from pickle...")
with open(OUTPUT_DIR / 'tsbs_k5_all-MiniLM-L6-v2_2005-2009_Jan04_22h.pkl', 'rb') as f:
    tsbs = pickle.load(f)

print(f"Loaded {len(tsbs)} TSBs")

# Deduplicate by ID (keep first occurrence)
print("\nDeduplicating TSBs...")
seen_ids = set()
unique_tsbs = []
for tsb in tsbs:
    if tsb.id not in seen_ids:
        seen_ids.add(tsb.id)
        unique_tsbs.append(tsb)

print(f"After dedup: {len(unique_tsbs)} unique TSBs ({len(tsbs) - len(unique_tsbs)} duplicates removed)")

print("\nRebuilding graph with unique TSBs...")
graph = build_graph(unique_tsbs, k=5)  # Use unique_tsbs instead of tsbs

print(f"\nNew graph has {len(graph)} nodes")

# Save the deduplicated TSBs
config = {
    'k': 5,
    'model_name': 'all-MiniLM-L6-v2',
    'dataset': '2005-2009',
    'num_tsbs': len(unique_tsbs)
}

print("\nSaving...")
save_graph_and_tsbs(unique_tsbs, graph, OUTPUT_DIR, config)  # Save unique_tsbs
