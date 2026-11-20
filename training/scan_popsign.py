import pyarrow.parquet as pq, fsspec, json, sys
from concurrent.futures import ThreadPoolExecutor
fs = fsspec.filesystem("https")
base = "https://huggingface.co/datasets/sign/popsign-images/resolve/main/data/"
def scan(name):
  try:
    return _scan(name)
  except Exception as e:
    return name, str(e)
def _scan(name):
    with fs.open(base + name, block_size=1<<20) as fh:
        f = pq.ParquetFile(fh)
        out = []
        for i in range(f.num_row_groups):
            st = f.metadata.row_group(i).column(3).statistics
            out.append((st.min, st.max, f.metadata.row_group(i).num_rows) if st and st.has_min_max else None)
        return name, out
    
import functools
names = [f"game/train-{i:05d}-of-00112.parquet" for i in range(111)] + [f"game/test-{i:05d}-of-00034.parquet" for i in range(34)]
with ThreadPoolExecutor(16) as ex:
    res = dict(ex.map(scan, names))
json.dump(res, open("index.json", "w"))
for n in names[:5]: print(n, res[n])
