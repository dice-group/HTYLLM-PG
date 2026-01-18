import os
import shutil

# 1. The list of languages
LANGUAGES = """
abk_Cyrl  azj_Cyrl  cpc_Latn  emp_Latn  gng_Latn  hnn_Latn  kgk_Latn  liv_Latn  mti_Latn  nya_Latn  pui_Latn  spm_Latn  tso_Latn  xav_Latn
acn_Latn  bbr_Latn  cpu_Latn  eng_Latn  gnw_Latn  hto_Latn  knv_Latn  lob_Latn  mww_Latn  ogo_Latn  pxm_Latn  srd_Latn  tue_Latn  xmm_Latn
aey_Latn  ben_Latn  crk_Latn  gqr_Latn  ibg_Latn  koi_Cyrl  lsi_Latn  myb_Latn  ojb_Cans  quz_Latn  srp_Cyrl  tuo_Latn  xnn_Latn
agg_Latn  bgs_Latn  crl_Cans  gso_Latn  ige_Latn  kpw_Latn  maf_Latn  nab_Latn  oku_Latn  rgu_Latn  swp_Latn  tzm_Tfng  yua_Latn
ake_Latn  bku_Latn  crx_Latn  guh_Latn  ile_Latn  kru_Deva  mau_Latn  naf_Latn  orv_Cyrl  rmn_Latn  syc_Syrc  ubu_Latn  yuj_Latn
amk_Latn  bon_Latn  csy_Latn  enx_Latn  gum_Latn  ium_Latn  ksc_Latn  mcb_Latn  naq_Latn  ory_Orya  rtm_Latn  tat_Latn  uvh_Latn  yut_Latn
amr_Latn  bov_Latn  ctp_Latn  esi_Latn  guu_Latn  iws_Latn  ksw_Mymr  mck_Latn  nas_Latn  ots_Latn  rus_Cyrl  tay_Latn  uzs_Arab  zpq_Latn
ang_Latn  bqp_Latn  cub_Latn  esu_Latn  gwr_Latn  jaa_Latn  kxm_Thai  mej_Latn  nba_Latn  otw_Latn  sah_Cyrl  tcy_Knda  wbm_Latn  zsm_Latn
arz_Arab  bug_Latn  cym_Latn  fas_Arab  gym_Latn  jav_Latn  kxw_Latn  mhr_Cyrl  ncj_Latn  pam_Latn  sat_Olck  tee_Latn  wew_Latn
asm_Beng  bwq_Latn  dhg_Latn  fin_Latn  hae_Latn  jbo_Latn  kyf_Latn  mie_Latn  nii_Latn  pma_Latn  sbd_Latn  tel_Latn  whg_Latn
atd_Latn  caa_Latn  dik_Latn  fuv_Latn  hbo_Hebr  jiv_Latn  kyu_Latn  miq_Latn  nin_Latn  pnt_Grek  sdh_Arab  tel_Telu  wlv_Latn
auc_Latn  cbi_Latn  dnj_Latn  gbi_Latn  hch_Latn  kaq_Latn  laj_Latn  mni_Beng  njo_Latn  poi_Latn  slv_Latn  tig_Ethi  wmt_Latn
avt_Latn  cbr_Latn  dty_Deva  gcf_Latn  hig_Latn  kat_Geor  lao_Laoo  moc_Latn  ntp_Latn  poy_Latn  sma_Latn  tkr_Cyrl  wsg_Telu
ayo_Latn  cho_Latn  dzo_Tibt  gej_Latn  hin_Deva  kck_Latn  lbb_Latn  mqj_Latn  nuj_Latn  prf_Latn  snd_Deva  toc_Latn  wuu_Hani
azg_Latn  chq_Latn  ekk_Latn  gmv_Latn  hin_Latn  keo_Latn  lgl_Latn  mse_Latn  nwx_Deva  prg_Latn  spa_Latn  toi_Latn  xal_Cyrl
"""

# 2. Configuration
OUTPUT_DIR = "custom_tasks/flores"
TASKS_FILE_PATH = "configs/lm_eval_tasks.txt"
SOURCE_LANG = "eng_Latn"
DATASET_PATH = "facebook/flores"
MAX_GEN_TOKS = 64 # TODO: check is this is enough for eval, or should we increase to 126

def clean_lang_list(raw_text):
    return sorted(list(set(raw_text.split())))

def generate_yaml(lang_code):
    if lang_code == SOURCE_LANG:
        return None
    
    task_name = f"flores_{SOURCE_LANG}-{lang_code}"
    dataset_name = f"{SOURCE_LANG}-{lang_code}"
    
    yaml_content = f"""task: {task_name}
dataset_path: {DATASET_PATH}
dataset_name: {dataset_name}
test_split: dev
dataset_kwargs:
  trust_remote_code: true
output_type: generate_until
doc_to_text: "{{{{sentence_{SOURCE_LANG}}}}} ="
doc_to_target: "{{{{sentence_{lang_code}}}}}"
metric_list:
  - metric: bleu
    aggregation: bleu
    higher_is_better: true
generation_kwargs:
  until:
    - "\\n"
  max_gen_toks: {MAX_GEN_TOKS}
metadata:
  version: 1.0
"""
    return task_name, yaml_content

def update_tasks_file(new_tasks, file_path):
    """Appends new tasks to the file if they aren't already there."""
    existing_tasks = set()
    
    # Read existing tasks to avoid duplicates
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            existing_tasks = set(line.strip() for line in f if line.strip())

    added_count = 0
    with open(file_path, 'a') as f:
        # If file was not empty and didn't end with newline, add one
        if existing_tasks and os.path.getsize(file_path) > 0:
             # This check is basic; 'a' mode writes to end. 
             # Ideally ensure newline separation.
             pass 

        for task in new_tasks:
            if task not in existing_tasks:
                f.write(f"\n{task}")
                added_count += 1
    
    return added_count

def main():
    # 1. Create directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    langs = clean_lang_list(LANGUAGES)
    print(f"Found {len(langs)} languages.")

    generated_tasks = []

    # 2. Generate YAMLs
    for lang in langs:
        res = generate_yaml(lang)
        if res:
            task_name, content = res
            file_path = os.path.join(OUTPUT_DIR, f"{task_name}.yaml")
            with open(file_path, "w") as f:
                f.write(content)
            generated_tasks.append(task_name)

    print(f"Generated {len(generated_tasks)} YAML files in '{OUTPUT_DIR}'.")

    # 3. Update the master tasks list file
    try:
        count = update_tasks_file(generated_tasks, TASKS_FILE_PATH)
        print(f"Successfully appended {count} new tasks to: {TASKS_FILE_PATH}")
    except Exception as e:
        print(f"Error updating tasks file: {e}")

if __name__ == "__main__":
    main()
