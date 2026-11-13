import json
import os
import statistics
from pathlib import Path
from collections import Counter, defaultdict

dev_path = Path("data/unified/dev.jsonl")
test_path = Path("data/unified/test_public.jsonl")
cuad_path = Path("data/raw/cuad/CUADv1.json")

def analyze():
    print("================ DATASET COMPOSITION ================")
    for path in [dev_path, test_path]:
        records = [json.loads(line) for line in path.open(encoding="utf-8")]
        print(f"\n--- {path.name} ---")
        print(f"Total documents: {len(records)}")
        
        source_counts = Counter(r["source_dataset"] for r in records)
        print(f"Sources: {dict(source_counts)}")
        
        total_fields = 0
        null_fields = 0
        present_fields = 0
        by_source = defaultdict(lambda: {"total": 0, "null": 0, "present": 0, "fields_per_doc": []})
        
        normalizers = Counter()
        types = Counter()
        tags_count = Counter()
        
        for r in records:
            src = r["source_dataset"]
            props = r.get("schema", {}).get("properties", {})
            gold = r["gold"]
            n_fields = len(gold)
            total_fields += n_fields
            by_source[src]["fields_per_doc"].append(n_fields)
            
            for t in r.get("tags", []):
                tags_count[t] += 1
                
            for f, g in gold.items():
                prop = props.get(f, {})
                norm = prop.get("x-normalizer", "string")
                ftype = prop.get("type", "string")
                if isinstance(ftype, list):
                    ftype = ",".join(ftype)
                normalizers[norm] += 1
                types[ftype] += 1
                
                by_source[src]["total"] += 1
                if g is None or (isinstance(g, str) and g.strip() == ""):
                    null_fields += 1
                    by_source[src]["null"] += 1
                else:
                    present_fields += 1
                    by_source[src]["present"] += 1
                    
        print(f"Total fields: {total_fields}, Present: {present_fields} ({present_fields/total_fields*100:.2f}%), Null: {null_fields} ({null_fields/total_fields*100:.2f}%)")
        for src, d in by_source.items():
            tot = d["total"]
            nl = d["null"]
            pr = d["present"]
            fpd = d["fields_per_doc"]
            print(f"  Source {src}: {tot} fields | Present: {pr} ({pr/tot*100:.2f}%) | Null: {nl} ({nl/tot*100:.2f}%) | Fields/doc: median={statistics.median(fpd):.1f}, mean={statistics.mean(fpd):.1f}, min={min(fpd)}, max={max(fpd)}")
            
        print("\nNormalizers:", dict(normalizers))
        print("Types:", dict(types))
        print("\nTop 15 Tags:", tags_count.most_common(15))

    # CUAD specific clause analysis
    print("\n================ CUAD CLAUSE ANALYSIS ================")
    dev_records = [json.loads(line) for line in dev_path.open(encoding="utf-8")]
    cuad_records = [r for r in dev_records if r["source_dataset"] == "cuad"]
    clause_stats = defaultdict(lambda: {"total": 0, "present": 0, "null": 0})
    for r in cuad_records:
        for f, g in r["gold"].items():
            clause_stats[f]["total"] += 1
            if g is not None and str(g).strip() != "":
                clause_stats[f]["present"] += 1
            else:
                clause_stats[f]["null"] += 1
                
    sorted_clauses = sorted(clause_stats.items(), key=lambda x: x[1]["present"] / x[1]["total"], reverse=True)
    print(f"Top 10 most common CUAD clauses (in dev, n={len(cuad_records)}):")
    for f, st in sorted_clauses[:10]:
        pct = st["present"] / st["total"] * 100
        print(f"  {f:35s}: {st['present']}/{st['total']} ({pct:.1f}% present)")
        
    print(f"\nTop 10 rarest CUAD clauses (in dev, n={len(cuad_records)}):")
    for f, st in sorted_clauses[-10:]:
        pct = st["present"] / st["total"] * 100
        print(f"  {f:35s}: {st['present']}/{st['total']} ({pct:.1f}% present)")

    # ExtractBench checkbox & boolean analysis
    print("\n================ EXTRACTBENCH FIELDS & BOOLEANS ================")
    eb_records = [r for r in dev_records if r["source_dataset"] == "extractbench"]
    bool_stats = Counter()
    for r in eb_records:
        props = r.get("schema", {}).get("properties", {})
        for f, g in r["gold"].items():
            ptype = props.get(f, {}).get("type")
            if ptype == "boolean":
                if g == "true":
                    bool_stats["true"] += 1
                elif g == "false":
                    bool_stats["false"] += 1
                elif g is None:
                    bool_stats["null"] += 1
                else:
                    bool_stats[f"other_{g}"] += 1
    print("Boolean/Checkbox value distribution in dev:", dict(bool_stats))

    # CUAD Document lengths
    print("\n================ DOCUMENT LENGTH & GEOMETRY ================")
    if cuad_path.exists():
        raw_cuad = json.loads(cuad_path.read_text(encoding="utf-8"))["data"]
        cuad_lengths = []
        for contract in raw_cuad:
            ctx = contract["paragraphs"][0]["context"]
            cuad_lengths.append(len(ctx))
        cuad_lengths.sort()
        n = len(cuad_lengths)
        print(f"CUAD Contracts total in raw: {n}")
        print(f"Length in chars: min={min(cuad_lengths)}, median={statistics.median(cuad_lengths):.0f}, mean={statistics.mean(cuad_lengths):.0f}, max={max(cuad_lengths)}")
        print(f"Percentiles: p10={cuad_lengths[int(n*0.10)]}, p50={cuad_lengths[int(n*0.50)]}, p75={cuad_lengths[int(n*0.75)]}, p90={cuad_lengths[int(n*0.90)]}, p95={cuad_lengths[int(n*0.95)]}, p99={cuad_lengths[int(n*0.99)]}")
        
        gt_60k = sum(1 for l in cuad_lengths if l > 60000)
        gt_90k = sum(1 for l in cuad_lengths if l > 90000)
        gt_180k = sum(1 for l in cuad_lengths if l > 180000)
        print(f"Contracts > 60,000 chars: {gt_60k}/{n} ({gt_60k/n*100:.1f}%)")
        print(f"Contracts > 90,000 chars: {gt_90k}/{n} ({gt_90k/n*100:.1f}%)")
        print(f"Contracts > 180,000 chars: {gt_180k}/{n} ({gt_180k/n*100:.1f}%)")

    # ExtractBench page counts & file sizes
    eb_pages = []
    for r in eb_records:
        for p in r.get("pages", []):
            full_p = Path("data") / p
            if full_p.exists():
                # check page count if pdf
                if full_p.suffix.lower() == ".pdf":
                    try:
                        import pypdf
                        reader = pypdf.PdfReader(full_p)
                        eb_pages.append(len(reader.pages))
                    except:
                        pass
    if eb_pages:
        print(f"\nExtractBench dev PDFs with page counts: {len(eb_pages)}")
        print(f"Pages per doc: min={min(eb_pages)}, median={statistics.median(eb_pages)}, mean={statistics.mean(eb_pages):.2f}, p90={sorted(eb_pages)[int(len(eb_pages)*0.9)]}, max={max(eb_pages)}")
        print(f"Page distribution:", Counter(eb_pages))

    # Tag analysis and perception slices
    print("\n================ TAG & PERCEPTION SLICE BREAKDOWN ================")
    tag_field_stats = defaultdict(lambda: {"docs": 0, "fields": 0, "null": 0, "present": 0})
    for r in dev_records:
        n_f = len(r["gold"])
        n_null = sum(1 for g in r["gold"].values() if g is None or (isinstance(g, str) and g.strip() == ""))
        n_pres = n_f - n_null
        for t in r.get("tags", []):
            tag_field_stats[t]["docs"] += 1
            tag_field_stats[t]["fields"] += n_f
            tag_field_stats[t]["null"] += n_null
            tag_field_stats[t]["present"] += n_pres
            
    print(f"{'Tag':30s} | {'Docs':5s} | {'Fields':6s} | {'Null%':6s} | {'Present%':8s}")
    print("-" * 65)
    for t in sorted(tag_field_stats.keys()):
        st = tag_field_stats[t]
        null_pct = st["null"] / st["fields"] * 100 if st["fields"] else 0
        pres_pct = st["present"] / st["fields"] * 100 if st["fields"] else 0
        print(f"{t:30s} | {st['docs']:5d} | {st['fields']:6d} | {null_pct:5.1f}% | {pres_pct:7.1f}%")

analyze()
