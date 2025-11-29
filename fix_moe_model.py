import os
import sys

def fix_modeling_file(model_dir):
    modeling_path = os.path.join(model_dir, "modeling_htyllm.py")
    if not os.path.exists(modeling_path):
        print(f"Error: {modeling_path} does not exist.")
        return False

    print(f"Patching {modeling_path}...")
    with open(modeling_path, "r") as f:
        content = f.read()

    old_code = """if not deepspeed.comm.is_initialized():
    deepspeed.init_distributed(dist_backend="nccl")"""

    new_code = """if not deepspeed.comm.is_initialized():
    import os
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    
    deepspeed.init_distributed(dist_backend="nccl", auto_mpi_discovery=False)"""

    if old_code in content:
        new_content = content.replace(old_code, new_code)
        with open(modeling_path, "w") as f:
            f.write(new_content)
        print("Successfully patched.")
        return True
    elif "auto_mpi_discovery=False" in content:
        print("File already patched.")
        return True
    else:
        print("Could not find the specific code block to patch. Please check the file manually.")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_moe_model.py <path_to_model_dir>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    fix_modeling_file(model_dir)

