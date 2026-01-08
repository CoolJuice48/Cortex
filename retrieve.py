"""
TSB Retrieval System

Provides functions to query the TSB graph and find relevant documents.

Last edited: 01/04/26
"""
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""

""" IMPORTS """
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from read_and_embed import TSB, load_graph_and_tsbs
""" END IMPORTS """

""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 Converts a text query into an embedding                        
 Returns an n-dimensional numpy array
>--=+=--<{--------------------}>--=+=--<
"""
def query_to_embedding(query: str, model: SentenceTransformer) -> np.ndarray:
   embedding = model.encode([query])[0]
   return embedding
""""""
"""      query_to_embedding()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{----------------------------}<--=+=--< 
 Finds TSBs most similar to the embedded query                        
 Returns a sorted list of (tsb_id, score) tuples
>--=+=--<{----------------------------}>--=+=--<
"""
def find_initial_matches(query_embedding: np.ndarray, tsbs: list[TSB],
                         top_k: int=5) -> list[tuple[str, float]]:
   # Cosine similarity
   # Normalized by model
   similarities = []
   seen_ids = set()

   for tsb in tsbs:
      # Skip duplicates
      if tsb.id in seen_ids:
         continue
      seen_ids.add(tsb.id)

      similarity = np.dot(query_embedding, tsb.embedding) / (
                   np.linalg.norm(query_embedding) * np.linalg.norm(tsb.embedding)
      )
      similarities.append((tsb.id, float(similarity)))

   # Sort by similarity (highest first) & return top k
   similarities.sort(key=lambda x: x[1], reverse=True)

   return similarities[:top_k]
""""""
"""      find_initial_matches()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{----------------------------}<--=+=--< 
 Looks for "nearby" nodes with highest scores                       
 Returns a list of adjacent nodes
>--=+=--<{----------------------------}>--=+=--<
"""
def explore_graph(starting_tsb_ids: list[str], graph: dict,
                      max_hops: int=2, top_k: int=20,
                      min_relevance: float=0.3) -> list[tuple[str, float]]:
   visited = {}

   # Initialize queue with relevant TSBs
   queue = []
   for tsb_id in starting_tsb_ids:
      if tsb_id in graph:
         queue.append((tsb_id, 1.0, 0))

   # BFS traversal
   while queue:
      curr_id, curr_weight, hops = queue.pop(0)

      # Skip if visited or too many degrees of separation
      if curr_id in visited or hops > max_hops:
         continue

      # Skip if irrelevant
      if curr_weight < min_relevance:
         continue

      # Mark as visited with score
      visited[curr_id] = curr_weight

      # Explore neighbors
      for nbr_id, score in graph.get(curr_id, []):
         # Decay relevance at increasing distances
         if nbr_id not in visited:
            new_weight = curr_weight * score
            # Only add if suitably relevant
            if new_weight >= min_relevance:
               queue.append((nbr_id, new_weight, hops+1))
         
   # Remove starting nodes
   for start_id in starting_tsb_ids:
      visited.pop(start_id, None)
   
   # Sort by relevance and return top k
   sorted_results = sorted(visited.items(), key=lambda x: x[1], reverse=True)

   return sorted_results[:top_k]
""""""
"""      explore_graph()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{----------------------------}<--=+=--< 
 Query -> relevant TSBs                       
 Returns a list of similar nodes
>--=+=--<{----------------------------}>--=+=--<
"""
def retrieve(query: str, 
             tsbs: list[TSB],
             graph: dict,
             model: SentenceTransformer,
             initial_k: int = 5,
             max_hops: int = 2,
             total_results: int = 20) -> list[tuple[TSB, float]]:
   
   print(f"\n{'='*60}")
   print(f"Query: '{query}'")
   print(f"{'='*60}")

   # Step 1: Convert query to embedding
   print("\n[1/4] Converting query to embedding...")
   query_embedding = query_to_embedding(query, model)

   # Step 2: Find initial matches via embedding similarity
   print(f"[2/4] Finding {initial_k} initial matches...")
   initial_matches = find_initial_matches(query_embedding, tsbs, top_k=initial_k)

   print(f"Found {len(initial_matches)} seed TSBs:")
   for tsb_id, score in initial_matches[:3]:
      print(f"  - {tsb_id}: {score:.3f}")

   # Step 3: Expand via graph traversal
   print(f"\n[3/4] Expanding via graph (max_hops={max_hops})...")
   starting_ids = [tsb_id for tsb_id, _ in initial_matches]
   expanded_matches = explore_graph(starting_ids, graph, max_hops, top_k=total_results)

   print(f"Found {len(expanded_matches)} related TSBs via graph")
   
   # Step 4: Combine and deduplicate results
   print(f"\n[4/4] Combining results...")
   
   # Create a map of tsb_id to TSB object for quick lookup
   tsb_map = {tsb.id: tsb for tsb in tsbs}
   
   # Combine initial matches and expanded matches
   all_results = {}
   
   # Add initial matches with their scores
   for tsb_id, score in initial_matches:
      all_results[tsb_id] = score
   
   # Add expanded matches (only if not already present or if score is higher)
   for tsb_id, score in expanded_matches:
      if tsb_id not in all_results or score > all_results[tsb_id]:
         all_results[tsb_id] = score
   
   # Sort by score and take top results
   sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
   top_results = sorted_results[:total_results]
   
   # Convert to (TSB object, score) tuples
   final_results = [(tsb_map[tsb_id], score) for tsb_id, score in top_results]
   
   print(f"✓ Returning {len(final_results)} results")
   print(f"{'='*60}\n")
   
   return final_results
""""""
"""      retrieve()      """
""""""
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{----------------------------}<--=+=--< 
 Nice display of results                     
 Returns nothing
>--=+=--<{----------------------------}>--=+=--<
"""
def display_results(results: list[tuple[TSB, float]], max_display: int = 10):
   print(f"\n{'='*80}")
   print(f"RETRIEVAL RESULTS ({len(results)} total)")
   print(f"{'='*80}\n")
   
   for i, (tsb, score) in enumerate(results[:max_display], 1):
      print(f"[{i}] Relevance: {score:.3f}")
      print(f"    ID: {tsb.id}")
      print(f"    Domain: {tsb.domain}")
      
      # Use .get() for dict access
      if 'make' in tsb.data:
         print(f"    Vehicle: {tsb.get('make')} {tsb.get('model')} {tsb.get('year')}")
         print(f"    Component: {tsb.get('component')}")
      
      print(f"    Summary: {tsb.get('summary', '')[:150]}...")
      print()
   
   if len(results) > max_display:
      print(f"... and {len(results) - max_display} more results\n")
""""""
"""      display_results()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
r"""
                                  __  __    _    ___ _   _ 
                                 |  \/  |  / \  |_ _| \ | |
                                 | |\/| | / _ \  | ||  \| |
                                 | |  | |/ ___ \ | || |\  |
                                 |_|  |_/_/   \_\___|_| \_|
                                                               
"""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
""""""
if __name__ == "__main__":
   from pathlib import Path
   
   OUTPUT_DIR = Path('/Volumes/256GB FLASH/Cortex/backend/embed/output')
   
   # Load your saved data
   print("Loading TSBs and graph...")
   version = "Jan05_21h_v006"  # 01/05/25 9:58PM
   tsbs, graph = load_graph_and_tsbs(OUTPUT_DIR, version)
   
   # Load model
   print("Loading model...")
   model = SentenceTransformer('all-MiniLM-L6-v2')
   
   # Test queries
   test_queries = [
      "engine won't start in cold weather",
      "brake pedal feels spongy",
      "transmission slipping",
      "battery dies overnight",
      "check engine light flashing"
   ]
   
   for query in test_queries:
      results = retrieve(query, tsbs, graph, model, 
                        initial_k=5, max_hops=2, total_results=10)
      display_results(results, max_display=5)
      
      input("\nPress Enter for next query...")