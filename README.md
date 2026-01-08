TSBS_STANDARD_V1
- Source: NHTSA TSBS flat files
- Rows: ~2.4M
- Required fields: manufacturer, model, model_year, component, summary

–––––––––––––––––––––––
       CHANGELOG
–––––––––––––––––––––––
01/04/26
   -  Wrote read_and_embed.py and retrieve.py
   -  Need to improve filtering in the "Finding neighors..." section
      of build_graph(), in read_and_embed.py(). Could be screwing up
      embeddings
   -  Considering hierarchical expert clusters with dynamic granularity
      (Holy shit)

   Domain Hierarchy Pseudocode:

   class Domain:
      ParentDomain // Pointer to
      ChildDomain // Pointer to

      array<Domains> similar_domains(self):
         similar = neighbors_of(self)
         return similar
   
   * What a domain is made up of: hierarchy matrix *
   #  matrix<Domains> consists_of(source: Domain):
         contents: matrix(rows=1, cols=1) // base case & memo, np.ndarray
         contents[middle] = source

         while (source.ChildDomain != nullptr):
            for sd in similar_domains(source.ChildDomain):
               contents[direction_of(source.ChildDomain)] = consists_of(source.ChildDomain)

         return contents

   #  list<Domains> similar_domains(source: Domain):
         for i in range(k):
            if cosine_similarity()

   #  MAIN
      query = input("Ask a question: ")
      embd_query = embed_query(query)
      initial = find_initial_matches(embd_query)
      similar = explore_graph(initial)

   array<Domains> X // Features of embedded data
   
   array<Domains> Y = X.similar_domains()
   array<Domains> subdomains = Y.consists_of()

   domains_in(y) = {first, second, third, ..., n} // List, or object of domains in a category ("Engine" in "Car", "Car" in "Vehicle")
   # Cortex
