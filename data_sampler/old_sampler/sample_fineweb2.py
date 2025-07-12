from multiprocessing import Pool, freeze_support
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter
import os

OUTPUT_DIR = "/data/adapter_fineweb2_subset/niger_congo" #Adapt
os.makedirs(OUTPUT_DIR, exist_ok=True)

language_codes = [ # 2Gb data -> 800.000.000 TOkens
        "swh_Latn", #1.3gb -> beleble_swh_Latn 
        "zul_Latn", # 182mb
        
        "nso_Latn", # 8mb -> belebele_nso_Latn
        "tsn_Latn", # 9mb -> beleble_tsn_Latn
        "sot_Latn", # 127mb ->
        
        "yor_Latn",# 96mb -
        "ibo_Latn", # 140mb
        
        "ewe_Latn", # 3mb -> no eval
        "fon_Latn", #2mb -> no eval
    ]

lang_code_2 = [
    "beng_Beng", #20gb ->  5gb 
    "asm_Beng",  # 334mb -> all
]

lang_code_2 = [
    "kaz_Cyrl", # 6,21gb
    "khk_Cyrl", # 2,5gb
]

lang_code_3 = [
    "amh_Ethi", # 530mb
    "tir_Ethi", # 141mb
]

lang_code_4 = [
	"ars_Arab", #1.81GB
	"apc_Arab",
	"arb_Arab", #94.52GB
	"acm_Arab",
	"arz_Arab"
]

def run_executor(lang):
    if lang not in language_codes:
        print(f"Skipping unknown language: {lang}")
        return
    reader_path = f"hf://datasets/HuggingFaceFW/fineweb-2/data/{lang}/train"
    output_path = os.path.join(OUTPUT_DIR, f"{lang}.jsonl")

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(reader_path),
            JsonlWriter(output_path)
        ],
        tasks=9
    )
    executor.run()

def main():
    max_parallel_languages = 9  # Adjust this to control parallelism

    with Pool(processes=max_parallel_languages) as pool:
        pool.map(run_executor, language_codes)

if __name__ == '__main__':
    freeze_support()
    main()
