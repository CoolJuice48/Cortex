
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 Data source formatting
 Columns are passed as 'data' param
>--=+=--<{--------------------}>--=+=--<
"""
class DataSchema:
   def __init__(self, 
               name: str,                  # Schema identifier
               domain: str,                # Domain/category
               columns: list[str],         # Ordered list of column names
               required_fields: list[str], # Fields that must be non-empty
               delimiter: str = '\t',      # How fields are separated in the file
               skip_values: dict = None):  # Fields and values indicative of bad data (e.g., "Unknown")

      self.name = name
      self.domain = domain
      self.columns = columns
      self.required_fields = required_fields
      self.delimiter = delimiter
      self.skip_values = skip_values or {}
   
   # Turns a line of data into a dict
   def parse_line(self, line: str) -> dict:
      fields = line.strip().split(self.delimiter)
      
      # Build dict
      data = {}
      for i, col_name in enumerate(self.columns):
         if i < len(fields):
               # Handle multi-field columns (like summary)
               if col_name == 'summary' and i < len(fields):
                  data[col_name] = self.delimiter.join(fields[i:])
                  break
               else:
                  data[col_name] = fields[i]
      
      return data
   # perse_line()

   # Determines whether data is well-formed
   def is_valid(self, data: dict) -> bool:
      """Check if data meets quality requirements."""
      # Check required fields exist and are non-empty
      for field in self.required_fields:
         if field not in data or not data[field]:
               return False
      
      # Check for skip values
      for field, skip_vals in self.skip_values.items():
         if field in data and data[field] in skip_vals:
               return False
      
      return True
   # is_valid()
   
   # Concats data fields for embedding
   def embedding_text(self, data: dict) -> str:
      """Generate text for embedding from this data."""
      # Default: concatenate all text fields
      text_fields = [str(v) for v in data.values() if isinstance(v, str) and v]
      return ' '.join(text_fields)
   # embedding_text()
""""""
"""      DataSchema      """
""""""
"""––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––"""
"""
>--=+=--<{--------------------}<--=+=--< 
 Outline for data type
 TODO: Determine which fields are best for standardization
>--=+=--<{--------------------}>--=+=--<
"""
NHTSA_TSB_SCHEMA = DataSchema(
   name="NHTSA_TSB",
   domain="automotive",
   columns=[
      "nhtsa_id", "tsb_number", "date_added", "entry_id",
      "date_released", "mfr_internal_id", "type", "make",
      "model", "year", "component", "mfr_component_system",
      "mfr_component_subsystem", "summary"
   ],
   required_fields=["make", "summary"],
   delimiter='\t',
   skip_values={
      'make': ['UNKNOWN', ''],
      'summary': ['']
   }
)
""""""
"""      NHTSA_TSB_SCHEMA      """
""""""
