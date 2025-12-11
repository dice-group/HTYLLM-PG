import pandas as pd
from itertools import product

models = [
    {"model_id": "llama3-1b", "family": "llama3", "size": "1B", "tokenizer": "default"},
    {"model_id": "llama3-3b", "family": "llama3", "size": "3B", "tokenizer": "default"},
    {"model_id": "llama3-8b", "family": "llama3", "size": "8B", "tokenizer": "default"},
    {"model_id": "llama3-8b-exttok", "family": "llama3", "size": "8B", "tokenizer": "extended"},
]

num_langs = [12, 95, 192]

def mk_subsets(prefix, source, generator, embedding):
    return [
        {
            "subset_id": f"{prefix}_{n}",
            "source": source,
            "generator": generator,
            "embedding": embedding,
            "num_langs": n,
        }
        for n in num_langs
    ]

lang_subsets = (
    mk_subsets("llm_flores_llama3", "flores", "llm", "llama3")
    + mk_subsets("llm_flores_glot500", "flores", "llm", "glot500")
    + mk_subsets("lang2vec_genetic", "lang2vec", "analysis", "genetic")
)

architectures = [{"arch_id": "hydralora"}, {"arch_id": "cola"}]
losses = [
    {"loss_id": "with_lang_prior", "lang_prior": True},
    {"loss_id": "no_lang_prior", "lang_prior": False},
]

rows = [
    {
        "model_id": m["model_id"],
        "model_family": m["family"],
        "model_size": m["size"],
        "tokenizer": m["tokenizer"],
        "subset_id": s["subset_id"],
        "subset_source": s["source"],
        "subset_generator": s["generator"],
        "subset_embedding": s["embedding"],
        "num_langs": s["num_langs"],
        "arch_id": a["arch_id"],
        "loss_id": l["loss_id"],
        "lang_prior": l["lang_prior"],
    }
    for m, s, a, l in product(models, lang_subsets, architectures, losses)
]

df = pd.DataFrame(rows)
df.to_csv("ablations_llm.csv", index=False)
df.to_excel("ablations_llm.xlsx", index=False)
print("Total configs:", len(df))
print(df.head())
