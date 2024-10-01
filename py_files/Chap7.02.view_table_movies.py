import lancedb
import pandas as pd

uri = "data/sample-lancedb"
db = lancedb.connect(uri)

table = "movies"

myTable=db.open_table(table)

records = myTable.to_pandas()

print(records)