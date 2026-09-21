from bewer import Dataset

ds = Dataset()
ds.add("the quick brown fox deleted test1 test2", "the quick red fox test1 test2 inserted")
example = ds[0]
alignment = example.metrics.levenshtein().alignment

print(alignment.num_matches)  # 3
print(alignment.num_substitutions)  # 1

for op in alignment:
    print(f"{op.type.name:10s} ref={op.ref!r} hyp={op.hyp!r}")

# Display a color-coded two-row alignment in the console (requires rich):
alignment.display()
