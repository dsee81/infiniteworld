"""
Infinite World - Action-Conditioned Video Generation Inference Script
======================================================================
A standalone inference script for generating long videos with action control.
"""

import sys
import os
import cv2
import math
import torch
import random
import json
import datetime
import importlib
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch.distributed as dist
import torchvision.transforms as transforms
import re

# Add original project root to path so unchanged modules are imported from the main repo.
VARIANT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(VARIANT_ROOT, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from infworld.utils.prepare_dataloader import get_obj_from_str
from infworld.utils.data_utils import (
    get_first_clip_from_video,
    get_last_clip_from_video,
    save_silent_video_overwrite,
)
from infworld.utils.dataset_utils import is_vid, is_img

# ============================================================================
# Action Mapping Dictionaries
# ============================================================================
MOVE_ACTION_MAP = {
    'no-op': 0,
    'go forward': 1,
    'go back': 2,
    'go left': 3,
    'go right': 4,
    'go forward and go left': 5,
    'go forward and go right': 6,
    'go back and go left': 7,
    'go back and go right': 8,
    'uncertain': 9
}

VIEW_ACTION_MAP = {
    'no-op': 0,
    'turn up': 1,
    'turn down': 2,
    'turn left': 3,
    'turn right': 4,
    'turn up and turn left': 5,
    'turn up and turn right': 6,
    'turn down and turn left': 7,
    'turn down and turn right': 8,
    'uncertain': 9
}

# ============================================================================
# Utility Functions
# ============================================================================
def extract_ckpt_step(path):
    """Extract checkpoint step number from path."""
    match = re.search(r'checkpoint-(\d+)\.ckpt', path)
    return int(match.group(1)) if match else 0

def resize_and_center_crop(image, target_size):
    """Resize image and center crop to target size."""
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    
    scale = max(target_h / orig_h, target_w / orig_w)
    final_h = math.ceil(scale * orig_h)
    final_w = math.ceil(scale * orig_w)
    
    resized = cv2.resize(image, (final_w, final_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized)[None, ...].permute(0, 3, 1, 2).contiguous()
    cropped = transforms.functional.center_crop(tensor, target_size)
    return cropped[:, :, None, :, :]  # [1, C, 1, H, W]

def setup_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def torch_gc():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def load_action_sequence(action_path):
    """Load action sequence from JSON file."""
    with open(action_path, 'r') as f:
        actions = json.load(f)
    
    move_indices = [MOVE_ACTION_MAP[a['move']] for a in actions]
    view_indices = [VIEW_ACTION_MAP[a['view']] for a in actions]
    return move_indices, view_indices

def _parse_cond_clip_len(value):
    """
    Parse INFWORLD_COND_CLIP_LEN.
    Accepts:
      - integer string >= 1 (e.g. "1", "16", "81")
      - "all" / "none" / "-1" to mean "use entire video" (clip_len=None)
    """
    if value is None:
        return 1
    v = str(value).strip().lower()
    if v in {"all", "none", "-1"}:
        return None
    try:
        n = int(v)
    except ValueError as e:
        raise ValueError(f"Invalid INFWORLD_COND_CLIP_LEN={value!r}. Use an int >= 1 or 'all'.") from e
    if n < 1:
        raise ValueError(f"Invalid INFWORLD_COND_CLIP_LEN={value!r}. Must be >= 1 or 'all'.")
    return n

def load_condition_image(image_path, bucket_config, cond_clip_len=None, cond_clip_mode=None):
    """Load and preprocess condition image."""
    if is_vid(image_path):
        # By default we condition on a single frame for efficiency.
        # Set INFWORLD_COND_CLIP_LEN (e.g. 16/32/81 or 'all') to use video context.
        if cond_clip_len is None:
            cond_clip_len = os.environ.get("INFWORLD_COND_CLIP_LEN", "1")
        if cond_clip_mode is None:
            cond_clip_mode = os.environ.get("INFWORLD_COND_CLIP_MODE", "first")

        clip_len = _parse_cond_clip_len(cond_clip_len)
        clip_mode = str(cond_clip_mode).strip().lower()
        if clip_mode not in {"first", "last", "uniform"}:
            raise ValueError("INFWORLD_COND_CLIP_MODE must be 'first', 'last', or 'uniform'")
        if clip_len is None:
            # "all" means full video regardless of clip_mode.
            frames = get_first_clip_from_video(image_path, clip_len=None)
        elif clip_mode == "uniform":
            cap = cv2.VideoCapture(image_path)
            all_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if all_frames <= 0:
                cap.release()
                raise ValueError(f"Unable to read video frame count for uniform sampling: {image_path}")
            indices = np.linspace(0, max(all_frames - 1, 0), num=min(clip_len, all_frames), dtype=int)
            frames = []
            wanted = set(int(i) for i in indices)
            idx = 0
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                if idx in wanted:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                idx += 1
            cap.release()
        elif clip_mode == "last":
            frames = get_last_clip_from_video(image_path, clip_len=clip_len)
        else:
            frames = get_first_clip_from_video(image_path, clip_len=clip_len)
    elif is_img(image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frames = [image]
    else:
        raise ValueError(f'Unsupported file format: {image_path}')
    
    processed_frames = []
    for frame in frames:
        ratio = frame.shape[0] / frame.shape[1]
        closest_bucket = sorted(bucket_config.keys(), key=lambda x: abs(float(x) - ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        
        tensor = resize_and_center_crop(frame, (target_h, target_w))
        tensor = (tensor / 255 - 0.5) * 2  # Normalize to [-1, 1]
        processed_frames.append(tensor)
    
    return torch.cat(processed_frames, dim=2)

# ============================================================================
# Distributed Setup (support single-GPU without torchrun to avoid port conflict)
# ============================================================================
def setup_distributed():
    """Setup distributed or single-GPU mode."""
    if 'RANK' in os.environ:
        # Launched by torchrun or similar
        rank = int(os.environ['RANK'])
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600*24))
        global_rank = dist.get_rank()
        num_processes = dist.get_world_size()
        return local_rank, global_rank, num_processes, True  # use_cp_init=True
    else:
        # Single process (no torchrun) - avoid port conflict, no dist init
        local_rank = 0
        global_rank = 0
        num_processes = 1
        torch.cuda.set_device(local_rank)
        return local_rank, global_rank, num_processes, False  # use_cp_init=False

local_rank, global_rank, num_processes, use_dist = setup_distributed()
print(f"[InfWorld] local_rank: {local_rank} | global_rank: {global_rank} | world_size: {num_processes}")

# Context parallel setup
context_parallel_size = int(os.environ.get("INFWORLD_CONTEXT_PARALLEL_SIZE", "1"))
if num_processes % context_parallel_size != 0:
    raise ValueError(
        f"INFWORLD_CONTEXT_PARALLEL_SIZE={context_parallel_size} must divide world_size={num_processes}"
    )
import infworld.context_parallel.context_parallel_util as cp_util
if use_dist:
    from infworld.context_parallel.context_parallel_util import init_context_parallel, get_dp_size, get_dp_rank, get_cp_rank
    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    dp_rank = get_dp_rank()
    dp_size = get_dp_size()
    cp_rank = get_cp_rank()
else:
    # Single process: set globals so get_dp_rank/get_dp_size work without dist
    cp_util.dp_rank = 0
    cp_util.dp_size = 1
    cp_util.cp_rank = 0
    cp_util.cp_size = 1
    dp_rank = 0
    dp_size = 1
    cp_rank = 0
enable_context_parallel = (context_parallel_size > 1)

# ============================================================================
# Configuration
# ============================================================================
# Inference settings
GLOBAL_SEED = int(os.environ.get("INFWORLD_SEED", "42"))
setup_seed(GLOBAL_SEED + global_rank)
SEED_PER_TASK = os.environ.get("INFWORLD_SEED_PER_TASK", "0") != "0"
SEED_TASK_STRIDE = int(os.environ.get("INFWORLD_SEED_TASK_STRIDE", "1000"))

def get_env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def get_env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


TEXT_CFG_SCALE = get_env_float("INFWORLD_TEXT_CFG_SCALE", 5.0)
NUM_SAMPLING_STEPS = get_env_int("INFWORLD_NUM_SAMPLING_STEPS", 30)
SHIFT = get_env_int("INFWORLD_SHIFT", 7)  # PX256: 3, PX627: 7, PX960: 11
NUM_CHUNKS = get_env_int("INFWORLD_NUM_CHUNKS", 13)  # Number of video chunks to generate
MAX_TASKS = get_env_int("INFWORLD_MAX_TASKS", 0) or None
HIGH_QUALITY_SAVE = os.environ.get("INFWORLD_HIGH_QUALITY_SAVE", "1") != "0"
COND_WINDOW_FRAMES = int(os.environ.get("INFWORLD_COND_WINDOW_FRAMES", "0"))  # 0 = use full buffer

# Paths - checkpoint_path is read from config (configs/infworld_config.yaml)
# Model config for this memory-fix variant
CONFIG_PATH = os.path.join(VARIANT_ROOT, 'infworld_config.yaml')

PROMPTS_YAML = os.environ.get(
    "INFWORLD_PROMPTS_YAML",
    os.path.join(PROJECT_ROOT, 'prompts', 'demo.yaml'),
)
BUCKET_CONFIG_NAME = 'ASPECT_RATIO_627_F64'

# Output directory
OUTPUT_BASE = os.environ.get("INFWORLD_OUTPUT_BASE", os.path.join(VARIANT_ROOT, 'outputs'))

# Negative prompt for generation quality
NEGATIVE_PROMPT = "many cars, crowds, Vivid hues, overexposed, static, blurry details, subtitles, style, work, artwork, image, still, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, motionless image, cluttered background, three legs, crowded background, walking backwards."

# ============================================================================
# Main Inference Loop
# ============================================================================
def resolve_path(path, root=PROJECT_ROOT):
    """Resolve path: if relative, join with project root."""
    if path is None:
        return path
    path = str(path).strip()
    if not os.path.isabs(path):
        path = os.path.join(root, path)
    return path


def load_dit_state_dict(checkpoint_path):
    """Load DiT state dict from .ckpt (torch) or .safetensors."""
    checkpoint_path = resolve_path(checkpoint_path)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    return state_dict


def load_prompt_entries(prompts_path):
    """Load and validate prompt entries from YAML."""
    cfg = OmegaConf.load(prompts_path)
    if "prompts" not in cfg:
        raise ValueError(f"Prompt YAML missing 'prompts': {prompts_path}")

    prompts = OmegaConf.to_container(cfg.prompts, resolve=True)
    validated = []
    for idx, entry in enumerate(prompts):
        # Supported formats:
        #  1) Legacy list: [prompt, cond_path, action_path] or [..., output_name]
        #  2) Dict: {prompt, cond_path/image_path, action_path, output_name?, cond_clip_len?, cond_clip_mode?}
        cond_clip_len = None
        cond_clip_mode = None
        action_start_mode = None
        if isinstance(entry, (list, tuple)) and len(entry) in (3, 4):
            prompt, image_path, action_path = entry[0], entry[1], entry[2]
            output_name = entry[3] if len(entry) == 4 else None
        elif isinstance(entry, dict):
            prompt = entry.get("prompt")
            image_path = entry.get("cond_path", entry.get("image_path"))
            action_path = entry.get("action_path")
            output_name = entry.get("output_name")
            cond_clip_len = entry.get("cond_clip_len")
            cond_clip_mode = entry.get("cond_clip_mode")
            action_start_mode = entry.get("action_start_mode")
        else:
            raise ValueError(
                f"Prompt entry {idx} must be a list [prompt, cond_path, action_path] "
                f"(+ optional output_name) or a mapping with keys "
                f"prompt/cond_path/action_path in {prompts_path}"
            )
        if not isinstance(prompt, str):
            raise ValueError(
                f"Prompt entry {idx} has non-string prompt {type(prompt).__name__}. "
                "Quote prompt text that contains ':' in YAML."
            )
        if not isinstance(image_path, str) or not isinstance(action_path, str):
            raise ValueError(
                f"Prompt entry {idx} image/action paths must be strings in {prompts_path}"
            )
        if output_name is not None and not isinstance(output_name, str):
            raise ValueError(
                f"Prompt entry {idx} output_name must be a string in {prompts_path}"
            )

        validated.append((prompt, image_path, action_path, output_name, cond_clip_len, cond_clip_mode, action_start_mode))

    return validated


def main():
    torch_gc()
    
    config_path = CONFIG_PATH
    args = OmegaConf.load(config_path)
    checkpoint_path = resolve_path(args.get("checkpoint_path", "checkpoints/models/diffusion_pytorch_model.safetensors"))
    
    ckpt_step = extract_ckpt_step(checkpoint_path)
    
    # Create output directory
    output_dir = os.path.join(OUTPUT_BASE, f"infworld-ckpt{ckpt_step}-step{NUM_SAMPLING_STEPS}-cfg{TEXT_CFG_SCALE}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[InfWorld] Loading checkpoint: {checkpoint_path}")
    print(f"[InfWorld] Config: {config_path}")
    print(f"[InfWorld] Output directory: {output_dir}")
    print(
        "[InfWorld] Inference settings: "
        f"steps={NUM_SAMPLING_STEPS}, chunks={NUM_CHUNKS}, "
        f"cfg={TEXT_CFG_SCALE}, max_tasks={MAX_TASKS or 'all'}"
    )
    print(
        "[InfWorld] Condition settings: "
        f"cond_clip_len={os.environ.get('INFWORLD_COND_CLIP_LEN', '1')}, "
        f"cond_clip_mode={os.environ.get('INFWORLD_COND_CLIP_MODE', 'first')}"
    )
    if COND_WINDOW_FRAMES > 0:
        print(f"[InfWorld] Condition window: last {COND_WINDOW_FRAMES} frames per chunk")
    
    # Resolve relative paths in config for models that load from disk
    if hasattr(args, "vae_cfg") and "vae_pth" in args.vae_cfg:
        args.vae_cfg.vae_pth = resolve_path(args.vae_cfg.vae_pth)
    if hasattr(args, "text_encoder_cfg"):
        if "checkpoint_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.checkpoint_path = resolve_path(args.text_encoder_cfg.checkpoint_path)
        if "tokenizer_path" in args.text_encoder_cfg:
            args.text_encoder_cfg.tokenizer_path = resolve_path(args.text_encoder_cfg.tokenizer_path)
    
    # Initialize models
    print("[InfWorld] Loading VAE...")
    vae = get_obj_from_str(args.vae_target)(**args.vae_cfg).to(local_rank)
    
    print("[InfWorld] Loading Text Encoder...")
    text_encoder = get_obj_from_str(args.text_encoder_target)(device=local_rank, **args.text_encoder_cfg)
    text_encoder.t5.model.to(local_rank)
    
    print("[InfWorld] Loading Scheduler...")
    scheduler = get_obj_from_str(args.scheduler_target)(**args.val_scheduler_cfg)
    scheduler.num_sampling_steps = NUM_SAMPLING_STEPS
    scheduler.shift = SHIFT
    
    print("[InfWorld] Loading DiT Model...")
    dtype = getattr(torch, args.amp_dtype)
    dit = get_obj_from_str(args.model_target)(
        out_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_context_parallel=enable_context_parallel,
        **args.model_cfg
    ).to(dtype)
    dit.eval()
    
    # Load DiT checkpoint (from config)
    state_dict = load_dit_state_dict(args.checkpoint_path)
    
    # Remove position embeddings (will be recomputed)
    state_dict.pop("pos_embed_temporal", None)
    state_dict.pop("pos_embed", None)
    
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    print(f"[InfWorld] Model loaded! Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    
    dit.to(local_rank)
    
    # Load bucket config
    from infworld.configs import bucket_config as bucket_config_module
    bucket_config = getattr(bucket_config_module, BUCKET_CONFIG_NAME)
    
    # Load prompts
    prompts_path = os.path.abspath(PROMPTS_YAML)
    target_prompts = load_prompt_entries(prompts_path)
    print(f"[InfWorld] Loaded {len(target_prompts)} prompts")
    
    # Process each prompt
    def _sanitize_output_name(name):
        # Keep filenames portable and predictable.
        name = str(name).strip().replace(" ", "_")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = name.strip("._-")
        return name or "output"

    default_action_start_mode = os.environ.get("INFWORLD_ACTION_START_MODE", "buffer").strip().lower()

    for task_idx, (prompt, image_path, action_path, output_name, task_cond_clip_len, task_cond_clip_mode, task_action_start_mode) in enumerate(target_prompts):
        if MAX_TASKS is not None and task_idx >= MAX_TASKS:
            break

        if task_idx % dp_size != dp_rank:
            continue

        # Optional: deterministic per-task reseed for easier large experiment suites.
        if SEED_PER_TASK:
            setup_seed(GLOBAL_SEED + global_rank + task_idx * SEED_TASK_STRIDE)
        
        if not os.path.exists(image_path):
            print(f"[InfWorld] Skipping task {task_idx}: Image not found - {image_path}")
            continue
        
        if not os.path.exists(action_path):
            print(f"[InfWorld] Skipping task {task_idx}: Action not found - {action_path}")
            continue
        
        print(f"[InfWorld] Task {task_idx}: {prompt[:50]}...")
        action_start_mode = (task_action_start_mode or default_action_start_mode).strip().lower()
        if action_start_mode not in {"buffer", "chunk_aligned"}:
            raise ValueError(
                f"Unsupported action_start_mode={action_start_mode!r}. Use 'buffer' or 'chunk_aligned'."
            )
        
        # Load condition image
        cond_video = load_condition_image(
            image_path,
            bucket_config,
            cond_clip_len=task_cond_clip_len,
            cond_clip_mode=task_cond_clip_mode,
        ).to(local_rank)
        
        with torch.no_grad():
            cond_latent = vae.encode(cond_video)
        
        # Load action sequence
        move_indices, view_indices = load_action_sequence(action_path)
        
        # Initialize video buffer
        video_buffer = cond_video.clone().cpu()
        
        # Latent size for generation
        latent_size = list(cond_latent.shape)
        latent_size[2] = 21  # Output frames per chunk
        latent_size = torch.Size(latent_size)
        
        # Generate video chunks
        for chunk_idx in range(NUM_CHUNKS):
            print(f"[InfWorld] Generating chunk {chunk_idx + 1}/{NUM_CHUNKS}")
            
            with torch.no_grad():
                # For long horizons, conditioning on the entire history can OOM.
                # Keep full history on CPU, but only feed the last N frames to the model.
                cond_for_encode = video_buffer
                if COND_WINDOW_FRAMES > 0 and video_buffer.shape[2] > COND_WINDOW_FRAMES:
                    cond_for_encode = video_buffer[:, :, -COND_WINDOW_FRAMES:, :, :]

                current_cond = cond_for_encode.to(local_rank)
                current_latent = vae.encode(current_cond)
            
            # Get action slice for current chunk
            if action_start_mode == "chunk_aligned":
                curr_start = chunk_idx * args.validation_data.num_frames
            else:
                curr_start = video_buffer.shape[2] - 1
            curr_end = curr_start + args.validation_data.num_frames
            
            move = torch.tensor(move_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            view = torch.tensor(view_indices[curr_start:curr_end], dtype=torch.long, device=local_rank)
            
            # Pad with the last specified action if the requested horizon extends
            # beyond the provided trajectory, instead of falling back to no-op.
            num_frames = args.validation_data.num_frames
            if move.shape[0] < num_frames:
                pad_len = num_frames - move.shape[0]
                last_move = move_indices[-1] if move_indices else MOVE_ACTION_MAP["no-op"]
                last_view = view_indices[-1] if view_indices else VIEW_ACTION_MAP["no-op"]
                move = torch.cat([move, torch.full((pad_len,), last_move, dtype=torch.long, device=local_rank)])
                view = torch.cat([view, torch.full((pad_len,), last_view, dtype=torch.long, device=local_rank)])
            
            additional_args = {
                "image_cond": current_latent,
                "move": move.unsqueeze(0),
                "view": view.unsqueeze(0),
            }
            
            torch_gc()
            
            with torch.no_grad():
                samples = scheduler.sample(
                    model=dit,
                    text_encoder=text_encoder,
                    null_embedder=dit.y_embedder,
                    z_size=latent_size,
                    prompts=[prompt],
                    guidance_scale=TEXT_CFG_SCALE,
                    negative_prompts=[NEGATIVE_PROMPT],
                    device=torch.device(local_rank),
                    additional_args=additional_args,
                )
                
                decoded_chunk = vae.decode(samples).cpu()
                video_buffer = torch.cat([video_buffer, decoded_chunk[:, :, 1:]], dim=2)
                
                print(f"[InfWorld] Chunk {chunk_idx + 1} done. Total frames: {video_buffer.shape[2]}")
                torch_gc()
        
        # Save final video
        if output_name:
            video_name = _sanitize_output_name(output_name)
        else:
            video_name = f"{task_idx:04d}_{prompt[:30].replace(' ', '_')}"
        save_path = os.path.join(output_dir, video_name)
        
        if cp_rank == 0:
            quality = 10 if HIGH_QUALITY_SAVE else 5
            save_silent_video_overwrite(
                video_buffer.to(local_rank),
                save_path,
                fps=30,
                quality=quality,
                high_quality_save=HIGH_QUALITY_SAVE,
            )
            print(f"[InfWorld] Saved: {save_path}.mp4")
        elif use_dist:
            print(f"[InfWorld] Skipping save on context-parallel rank {cp_rank}")

        if use_dist:
            dist.barrier()

if __name__ == "__main__":
    main()
