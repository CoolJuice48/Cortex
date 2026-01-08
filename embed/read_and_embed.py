""""""
"""
>----------------------------------------<
Docstring for backend.embed.read_and_embed

TSB processing and embedding

FOR TEST SAMPLE: 50k TSBS

Last edited: 01/05/26
>----------------------------------------<
"""
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""

""" IMPORTS """
import numpy as np
import pickle
import time
from functools import wraps
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from schemas import DataSchema, NHTSA_TSB_SCHEMA
""" END IMPORTS """

""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 Timer decorator & function                        
 Tracks function execution time
>--=+=--<{--------------------}>--=+=--<
"""
def timer(func):
   @wraps(func)
   def wrapper(*args, **kwargs):
      start = time.time()
      result = func(*args, **kwargs)
      elapsed = time.time() - start
      
      print(f"{func.__name__} completed in {elapsed:.2f}s")
      return result
   return wrapper
""""""
"""      Timer      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 TSB object                        
 Represents a single TSB entry
>--=+=--<{--------------------}>--=+=--<
"""
class TSB:
   # To expand to new kinds of data, include a 'domain' tag
   # These tags are fundamental to the model; their design is
   # necessary to consider when increasing modularity
   # 01/05/26, 3:21 AM: Wise words. Now using ['id', 'embedding', 'data', 'domain']

   __slots__ = ['id', 'embedding', 'data', 'domain']

   _next_id = 0

   def __init__(self, data: dict, domain: str=None):
      self.id = TSB._next_id
      TSB._next_id += 1

      self.data = data
      self.domain = domain
      self.embedding = None

   # Data getter
   def get(self, field: str, default=None):
        return self.data.get(field, default)
   
   def __repr__(self):
      preview = {k: v for k, v in list(self.data.items())[:3]}
      return f"TSB(id={self.id}, domain={self.domain}, data={preview}...)"
""""""
"""      TSB      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 Loads a TSB file & embeds in batches                      
 Returns a list of embedded TSBs
>--=+=--<{--------------------}>--=+=--<
"""
@timer
def load_and_embed(filepath: str,
                   schema: DataSchema,
                    model: SentenceTransformer,
                    batch_size: int=1000) -> list[TSB]:
   tsbs  = []
   texts = []
   TSB._next_id = 0

   with open(filepath, 'r') as f:
      total_lines = sum(1 for _ in f)

   with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
      for line in tqdm(f, total=total_lines, desc=f"Loading {schema.name}..."):

         # Schema line parsing function
         data = schema.parse_line(line)

         # Skip entry if malformed
         if not schema.is_valid(data):
            continue

         # Create TSB
         tsb = TSB(data=data, domain=schema.domain)
         tsbs.append(tsb)
         texts.append(schema.embedding_text(data))

         # Embed in batches of batch_size
         if len(texts) >= batch_size:
            embeddings = model.encode(texts, show_progress_bar=False)
            for tsb, emb in zip(tsbs[-batch_size:], embeddings):
               tsb.embedding = emb
            texts = []

   # Embed remaining
   if texts:
      embeddings = model.encode(texts, show_progress_bar=False)
      for tsb, emb in zip(tsbs[-len(texts):], embeddings):
         tsb.embedding = emb
      texts = []

   print(f"\nFiltering bad embeddings...")
   valid_tsbs = []
   for tsb in tsbs:
    if tsb.embedding is not None:
      if np.any(tsb.embedding != 0) and not np.any(np.isnan(tsb.embedding)):
         valid_tsbs.append(tsb)

   print(f"\nFiltered:{len(tsbs)} -> {len(valid_tsbs)} TSBs ({len(tsbs) - len(valid_tsbs)} removed)")
   
   return valid_tsbs
""""""
"""      load_and_embed()      """
""""""
"""–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""                                 WEIGHTED GRAPH CONSTRUCTION                                 """
"""
>--=+=--<{------------------------------------------------}<--=+=--<  
 Each TSB connects to its k most similar TSBS                             
 Returns a dict of tsb_id -> (neighbor_id, similarity)
>--=+=--<{------------------------------------------------}>--=+=--<  
"""
@timer
def build_graph(tsbs: list[TSB], k: int=5) -> dict:
   n = len(tsbs)

   # Early exit
   if n == 0:
      return {}

   print(f"\n[1/3]: Building k-NN graph -- k={k}, {n} TSBSs processing...")
   embeddings = np.array([tsb.embedding for tsb in tsbs])

   # Normalize embeddings (make each vector unit length)
   print("Normalizing embeddings...")
   norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
   norms[norms == 0] = 1  # Avoid division by zero
   embeddings = embeddings / norms

   print(f"\n[2/3]: Finding neighbors...")
   nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto', metric='cosine')
   nbrs.fit(embeddings)
   distances, indices = nbrs.kneighbors(embeddings)

   # Construct adjacent list
   print(f"\n[3/3]: Building graph...")
   graph = {tsb.id: [] for tsb in tsbs}

   for i, tsb in enumerate(tqdm(tsbs, desc="Building edges...")):
      neighbors = []
      for j in range(1, k+1):
         nbr_idx = indices[i][j]
         nbr_id  = tsbs[nbr_idx].id
         score   = 1.0 - distances[i][j]
         neighbors.append((nbr_id, score))

      graph[tsb.id] = neighbors

   print(f"\nGraph built: {n} nodes, {n * k} edges")

   return graph
""""""
"""      build_graph()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{------------------------------------------------}<--=+=--<  
 Saves graph and TSBs with text embeddings
 Returns nothing
>--=+=--<{------------------------------------------------}>--=+=--<  
"""
@timer
def save_graph_and_tsbs(tsbs: list[TSB], graph: dict, output_dir: Path, config: dict) -> str:
   output_dir.mkdir(parents=True, exist_ok=True)

   # Get next version number
   existing_versions = list(output_dir.glob('tsbs_*.pkl'))
   version_num = len(existing_versions) + 1
   
   # Build simple version string
   timestamp = datetime.now().strftime("%b%d_%Hh")

   version = f"{timestamp}_v{version_num:03d}"

   tsb_file = output_dir / f'tsbs_{version}.pkl'
   graph_file = output_dir / f'graph_{version}.pkl'
   config_file = output_dir / f'config_{version}.txt'

   # Save TSBs
   with open(tsb_file, 'wb') as f:
      pickle.dump(tsbs, f)
   print(f"\nSaved {len(tsbs)} TSBs")

   # Save graph
   with open(graph_file, 'wb') as f:
      pickle.dump(graph, f)
   print(f"\nSaved graph with {len(graph)} nodes")

   # Save config with all metadata
   with open(config_file, 'w') as f:
      f.write(f"Version: {version}\n")
      f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
      f.write(f"=" * 50 + "\n")
      for key, value in config.items():
         f.write(f"{key}: {value}\n")
   
   print(f"\nVersion {version} saved.")
   return version
""""""
"""      save_graph_and_tsbs()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{------------------------------------------------}<--=+=--<  
 Loads graph and TSBs                            
 Returns embedded TSBs and graph
>--=+=--<{------------------------------------------------}>--=+=--<  
"""
def load_graph_and_tsbs(output_dir: Path, version: str) -> tuple[list[TSB], dict]:
   tsb_file = output_dir / f'tsbs_{version}.pkl'
   graph_file = output_dir / f'graph_{version}.pkl'

   if not tsb_file.exists():
      raise FileNotFoundError(f"No saved data for version: {version}")

   # Load TSBs
   with open(tsb_file, 'rb') as f:
      tsbs = pickle.load(f)

   # Load graph
   with open(graph_file, 'rb') as f:
      graph = pickle.load(f)

   print(f"\nLoaded version {version}")
   print(f"\n{len(tsbs)} TSBs and graph with {len(graph)} nodes")

   return tsbs, graph
""""""
"""      load_graph_and_tsbs()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{------------------------------------------------}<--=+=--<  
 Lists available program versions                            
 Returns a string containing the version
>--=+=--<{------------------------------------------------}>--=+=--<  
"""
def list_versions(output_dir: Path) -> list[str]:
   # Identify and store existing versions
   tsb_files = list(output_dir.glob('tsbs_*.pkl'))
   versions = [f.stem.replace('tsbs_', '') for f in tsb_files]

   print(f"\nAvailable versions ({len(versions)}):")
   for v in sorted(versions):
      print(f"    - {v}")

   return versions
""""""
"""      list_versions()      """
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
   """--------------------------------"""
   """          PATH TO FILE          """
   OUTPUT_DIR = Path('/Volumes/256GB FLASH/Cortex/backend/embed/output')
   TSB_DIR = Path('/Volumes/256GB FLASH/Cortex/data/NHTSA_extracted/TSB_txts')
   TEST_FILE = 'TSBS_RECEIVED_2005-2009.txt'
   FILEPATH = TSB_DIR / TEST_FILE
   """^^^^^ LAST EDITED: 01/05/26 ^^^^^"""
   """"""
   
   """   WEIGHTED GRAPH CONFIGURATION  """
   """  EDIT TO TUNE MODEL PERFORMANCE """
   VERSION = {
      'k': 5,                           # Number of NearestNeighbors to look for
      'model_name': 'all-MiniLM-L6-v2', # SentenceTransformer embedding model
      'dataset': '2005-2009',           # Data to embed & graph (CURRENTLY A TEST FILE)
      'batch_size': 1000                # Number of parallel computations to run
   }
   """^^^^^ LAST EDITED: 01/05/26 ^^^^^"""
   """"""
   
   

   # Check for existing versions
   versions = list_versions(OUTPUT_DIR)

   # Either load existing or process new
   if versions:
      print("\nOptions:")
      print("1. Load existing version")
      print("2. Create new version (alter version config)")
      choice = input("Choice (1/2): ")

      if choice == "1": # Existing version selection
         print("\nAvailable versions:")
         for i, v in enumerate(versions, 1):
            print(f"{i}. {v}")
         idx = int(input("Select version: ")) - 1
         tsbs, graph = load_graph_and_tsbs(OUTPUT_DIR, versions[idx])
      
      else: # New data processed from FILEPATH configuration
         print("Processing new data...")
         print("Loading model...")
         model = SentenceTransformer(VERSION['model_name'])
         print(f"\nProcessing: {TEST_FILE}...")
         tsbs = load_and_embed(FILEPATH, NHTSA_TSB_SCHEMA, model, batch_size=VERSION['batch_size'])
         print(f"\nBuilding graph...")
         graph = build_graph(tsbs, k=VERSION['k'])
         print(f"\nSaving embedded TSBs and graph to disc...")
         version = save_graph_and_tsbs(tsbs, graph, OUTPUT_DIR, VERSION)
   
   # No versions found
   else: # New data processed from FILEPATH configuration
      print("No existing versions found. Creating new...")
      print("Loading model...")
      model = SentenceTransformer(VERSION['model_name'])
      print(f"\nProcessing: {TEST_FILE}...")
      tsbs = load_and_embed(FILEPATH, NHTSA_TSB_SCHEMA, model, batch_size=VERSION['batch_size'])
      print(f"\nBuilding graph...")
      graph = build_graph(tsbs, k=VERSION['k'])
      print(f"\nSaving embedded TSBs and graph to disc...")
      version = save_graph_and_tsbs(tsbs, graph, OUTPUT_DIR, VERSION)

   # print(f"\nSample graph entry:")
   # sample_id = tsbs[0].id
   # print(f"    TSB {sample_id} neighbors:")
   # for nbr_id, score in graph[sample_id]:
   #    print(f"    -> {nbr_id}: {score:.3f}")
