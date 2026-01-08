""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""

""" IMPORTS """
import json
import numpy as np
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from read_and_embed import TSB
from typing import Any, Protocol
""" END IMPORTS """

""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{------------------------------}<--=+=--< 
 General document format                        
 Domain-agnostic
>--=+=--<{------------------------------}>--=+=--<
"""
class EmbeddedDocument(Protocol):
   embedding: np.ndarray
   data: dict
   id: int
   domain: str
""""""
"""      EmbeddedDocument      """
""""""
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{------------------------------}<--=+=--< 
 Cluster data based on embeddings                        
 Returns a dict mapping cluster_id -> list of data
>--=+=--<{------------------------------}>--=+=--<
"""
def cluster(docs: list[Any], num_clusters: int=10, method: str='kmeans') -> dict:
   print(f"\nClustering {len(docs)} nodes into {num_clusters} clusters...")

   # Extract embeddings from list
   embeds = np.array([doc.embedding for doc in docs])

   # KMeans clustering
   kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
   cluster_labels = kmeans.fit_predict(embeds)

   # Calculate silhouette score (quality measurement)
   silhouette = silhouette_score(embeds, cluster_labels)
   print(f"Silhouette score: {silhouette:.3f} (higher is better, -1 to 1)")

   # Group datapoints by cluster
   clusters = {}
   for doc, label in zip(docs, cluster_labels):
      if label not in clusters:
         clusters[label] = []
      clusters[label].append(doc)
   
   print("\nCluster sizes:")
   for cluster_id in sorted(clusters.keys()):
      print(f"  Cluster {cluster_id}: {len(clusters[cluster_id])} TSBs")
   
   return clusters
""""""
"""      cluster()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{-------------------------------------}<--=+=--< 
 Gather common data of an embedded cluster                       
 Returns a dict with a cluster sample & frequent metadata
>--=+=--<{-------------------------------------}>--=+=--<
"""
def analyze_cluster(docs: list[Any], key_fields: list[str]=None, sample_size: int=10) -> dict:
   # Sample of cluster data, taken from middle
   sorted_docs = sorted(docs, key=lambda d: str(d.data))
   step = max(1, len(sorted_docs) // sample_size)
   sample = [sorted_docs[i] for i in range(0, len(sorted_docs), step)][:sample_size]

   # Analyze important fields
   field_analysis = {}
   if key_fields:
      for field in key_fields:
         values = [doc.data.get(field, 'Unknown') for doc in docs if field in doc.data]
         if values:
            field_analysis[field] = Counter(values).most_common(5)
   return {
      'sample': sample,
      'field_analysis': field_analysis,
      'size': len(docs)
   }
""""""
"""      analyze_cluster()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{-------------------------------------}<--=+=--< 
 Determine domain of an embedded cluster                       
 Returns the domain name as a string
>--=+=--<{-------------------------------------}>--=+=--<
"""
def name_cluster(analyzed_cluster: dict, domain_context: str=None) -> str:
   import anthropic

   # Build prompt from analyzed cluster
   prompt = build_naming_prompt(analyzed_cluster, domain_context)

   # Call API
   client = anthropic.Anthropic()
   message = client.message.create(
      model="claude-sonnet-4-20250514",
      max_tokens=50,
      messages=[{"role": "user", "content": prompt}]
   )

   domain_name = message.content[0].text.strip()
   return domain_name
""""""
"""      name_cluster()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{-------------------------------------}<--=+=--< 
 Build prompt for an API to name a cluster                       
 Returns the prompt as a string
>--=+=--<{-------------------------------------}>--=+=--<
"""
def build_naming_prompt(analyzed_cluster: dict, domain_context: str=None) -> str:
   # Get sample data
   samples = analyzed_cluster['sample_docs']
   sample_text = "\n".join([
      f"- {str(doc.data)[:150]}"
      for doc in samples[:5]
   ])
   
   # Get field analysis
   field_text = ""
   for field, counts in analyzed_cluster['field_analysis'].items():
      top_values = ", ".join([f"{val} ({count})" for val, count in counts[:3]])
      field_text += f"\n{field}: {top_values}"
   
   context_line = f"The data is: {domain_context}\n" if domain_context else ""
   
   prompt = f"""You are analyzing a cluster of {analyzed_cluster['size']} documents.

{context_line}
Common field values in this cluster:
{field_text}

Sample documents from this cluster:
{sample_text}

Based on this information, provide a concise 2-4 word domain name that describes what this cluster represents.
The name should be general enough to cover all items but specific enough to be meaningful.

Respond with ONLY the domain name, nothing else."""

   return prompt
""""""
"""      build_naming_prompt()      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{---------------------------------------------------------------------------}<--=+=--< 
 Pipeline: clusters docs & assigns domains                       
 Returns a tuple of two dicts: 1) domain_name -> list(docs), 2) cluster_info -> domain metadata
>--=+=--<{---------------------------------------------------------------------------}>--=+=--<
"""
def assign_domains(documents: list[Any], 
               num_clusters: int = 10,
               key_fields: list[str] = None,
               domain_context: str = None) -> tuple[dict, dict]:

   print("="*60)
   print("AUTOMATIC DOMAIN DISCOVERY")
   print("="*60)
   
   # Step 1: Cluster
   clusters = cluster(documents, num_clusters=num_clusters)
   
   # Step 2: Analyze and name each cluster
   print("\nNaming clusters...")
   domains = {}
   cluster_info = {}
   
   for cluster_id, cluster_docs in clusters.items():
      print(f"\nCluster {cluster_id}:")
      
      # Analyze
      analysis = analyze_cluster(cluster_docs, key_fields=key_fields)
      analysis['cluster_id'] = cluster_id
      
      print(f"  Size: {analysis['size']}")
      if analysis['field_analysis']:
         for field, counts in analysis['field_analysis'].items():
               print(f"  Top {field}: {[val for val, _ in counts[:3]]}")
      
      # Name
      domain_name = name_cluster(analysis, domain_context)
      print(f"  → Domain name: {domain_name}")
      
      # Store
      domains[domain_name] = cluster_docs
      cluster_info[domain_name] = {
         'cluster_id': cluster_id,
         'size': analysis['size'],
         'field_analysis': analysis['field_analysis']
      }
   
   print("\n" + "="*60)
   print("DOMAIN DISCOVERY COMPLETE")
   print("="*60)
   
   return (domains, cluster_info)
""""""
"""      assign_domains()      """
""""""


def save_domain_assignments(domains: dict, 
                           cluster_info: dict, 
                           output_path: str,
                           sample_fields: list[str] = None):
   """Save domain assignments to JSON for inspection."""
   output = {}
   
   for domain_name, docs in domains.items():
      # Get sample documents
      samples = []
      for doc in docs[:5]:
         sample = {'id': doc.id}
         
         # Include specified fields or all fields
         if sample_fields:
               for field in sample_fields:
                  sample[field] = doc.data.get(field, None)
         else:
               sample['data'] = {k: str(v)[:100] for k, v in list(doc.data.items())[:5]}
         
         samples.append(sample)
      
      output[domain_name] = {
         'info': cluster_info[domain_name],
         'document_ids': [doc.id for doc in docs],
         'sample_documents': samples
      }
   
   with open(output_path, 'w') as f:
      json.dump(output, f, indent=2)
   
   print(f"\nDomain assignments saved to: {output_path}")


# ============== Testing / Main ==============

if __name__ == "__main__":
   from pathlib import Path
   import sys
   
   # Add parent directory to path for imports
   sys.path.append(str(Path(__file__).parent.parent))
   
   from embed.read_and_embed import load_graph_and_tsbs
   
   # Load existing embedded documents
   OUTPUT_DIR = Path('/Volumes/256GB FLASH/Cortex/backend/embed/output')
   VERSION = 'Jan05_21h_v006'
   
   print("Loading documents...")
   documents, graph = load_graph_and_tsbs(OUTPUT_DIR, VERSION)
   print(f"Loaded {len(documents)} documents")
   
   # Run domain discovery
   # For TSBs specifically, we analyze these fields:
   domains, cluster_info = assign_domains(
      documents, 
      num_clusters=10,
      key_fields=['component', 'make'],  # TSB-specific fields
      domain_context="automotive technical service bulletins",
      use_llm=False  # Set True to use API
   )
   
   # Save results
   save_domain_assignments(
      domains, 
      cluster_info,
      OUTPUT_DIR / 'domain_assignments.json',
      sample_fields=['make', 'model', 'component', 'summary']
   )
   
   # Print summary
   print("\n" + "="*60)
   print("DISCOVERED DOMAINS:")
   print("="*60)
   for domain_name in sorted(domains.keys()):
      size = len(domains[domain_name])
      print(f"{domain_name:30s} {size:6d} documents")
