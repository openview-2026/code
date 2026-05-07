from utils.vllm_agent import build_vllm_agent

import os, yaml, cv2, argparse
from pathlib import Path
from typing import List
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils.utils import (
    extract_views, pose2uv, read_variable_saves, save_variable
)
from utils.variables import ViewMeta, ViewCaption, PanoSummary
from utils.parser import parser


def load_config(path):
    with open(path, "r") as f: 
        return yaml.safe_load(f)

def remove_patch_views(batch_out_dir):
    for out_dir in batch_out_dir:
        for file in os.listdir(out_dir):
            if file.endswith(".jpg") or file.endswith(".png"):
                os.remove(os.path.join(out_dir, file))

def process_pano(pano_info, cfg, backend, overwrite=False):
    # Stage 2: sample views and captioning
    pano_path, pano_label = pano_info

    # get pano info
    pano_id = Path(pano_path).stem
    cols = cfg["patch_cols"]
    rows = cfg["patch_rows"]
    patch_prefix = f"{cols}x{rows}"
    
    # overwrite with cache directory if exists
    cache_dir = cfg.get("cache_dir", None)
    if cache_dir:
        pinhole_out_dir = Path(cache_dir) / pano_id / patch_prefix
    else:
        pinhole_out_dir = Path(cfg["out_dir"]) / pano_id / patch_prefix
    out_dir = pinhole_out_dir / backend
    os.makedirs(out_dir, exist_ok=True)

    # read or generate summary
    if not overwrite:
        captions = read_variable_saves(out_dir, "captions")
        if captions: 
            try:
                pano_summary = read_variable_saves(out_dir, "summary")
            except Exception as e:
                print(f"[ERROR] Cannot read summary, generating summaries for {pano_path}, error: {e}")
                
            if pano_summary: 
                print(f"[INFO] Summary loaded from cache: {out_dir}")
                remove_patch_views([str(Path(out_dir).parent)])
                return None

    # sample views
    views_meta: List[ViewMeta] = []
    pano_img = cv2.imread(pano_path, cv2.IMREAD_COLOR)
    assert pano_img is not None, f"Failed to read pano: {pano_path}"
    meta = extract_views(pano_img, cols, rows, save_dir=pinhole_out_dir)

    for pose in meta["poses"]:
        views_meta.append(ViewMeta(
            view_id=pose["image_path"].stem,
            yaw_deg=float(pose["yaw"]),
            pitch_deg=float(pose["pitch"]),
            roll_deg=float(pose["roll"]),
            fov_x_deg=float(meta["fov_degs"][0]),
            fov_y_deg=float(meta["fov_degs"][1]),
            fov_diag_deg=float(meta["fov_degs"][2]),
            aspect_ratio=float(meta["out_size"][1] / meta["out_size"][0]),
            image_path=str(pose["image_path"]),
            left_neighbor_view=str(pose["left_neighbor_view"]),
            right_neighbor_view=str(pose["right_neighbor_view"]),
            top_neighbor_view=str(pose["top_neighbor_view"]),
            bottom_neighbor_view=str(pose["bottom_neighbor_view"]),
        ))

    # read or generate captions
    if not overwrite and captions: 
        print(f"[INFO] Generating summaries for {pano_path}")
        panorama_size = pano_img.shape[:2]
        return captions, out_dir, (pano_path, pano_label, panorama_size)

    print(f"[INFO] Generating captions and summaries for {pano_path}")
    return views_meta, out_dir, (pano_path, pano_label, meta["pano_size"])

def infer_and_flush(buffer, agent):
    if not buffer:
        return
    batch_item_0     = [it[0] for it in buffer]
    batch_out_dir    = [it[1] for it in buffer]
    batch_pano_path  = [it[2][0] for it in buffer]
    batch_pano_label = [it[2][1] for it in buffer]
    batch_pano_size  = [it[2][2] for it in buffer]

    # gather all unfinised items
    uncompleted = []
    for idx, item in enumerate(batch_item_0):
        if isinstance(item[0], ViewMeta):
            paths = [vm.image_path for vm in item]
            uncompleted.append({
                'idx': idx,
                'item': item,
                'paths': paths,
                'pano_size': batch_pano_size[idx],
                'out_dir': batch_out_dir[idx],
            })
    batch_captions = [None] * len(batch_item_0)

    if uncompleted:
        all_path = [p for u in uncompleted for p in u['paths']]
        all_caps = agent.caption(all_path)

        ptr = 0
        for u in uncompleted:
            item = u['item']
            n = len(item)
            caps_slice = all_caps[ptr:ptr+n]
            ptr += n

            captions = []
            for vm, cap in zip(item, caps_slice):
                # format handler
                if isinstance(cap, list):
                    cap = cap[0]
                if not isinstance(cap, dict):
                    cap = {"caption": cap}
                    print(f"[WARNING] Caption not in dict format in {u['out_dir']}")

                u_norm, v_norm = pose2uv(vm.yaw_deg, vm.pitch_deg)
                _caption = cap.get("caption", "")
                _objects = cap.get("objects", [])
                _spatial_facts = cap.get("spatial_facts", [])

                if isinstance(_caption, dict):
                    print(f"[WARNING] Caption not in dict format in {u['out_dir']}")
                    temp_caption = ""
                    for c in _caption.values():
                        if isinstance(c, list):
                            temp_caption.extend(c)
                        else:
                            temp_caption.append(c)
                    _caption = temp_caption
                if isinstance(_objects, dict):
                    print(f"[WARNING] Objects not in dict format in {u['out_dir']}")
                    temp_objects = []
                    for o in _objects.values():
                        if isinstance(o, list):
                            temp_objects.extend(o)
                        else:
                            temp_objects.append(o)
                    _objects = temp_objects
                if isinstance(_spatial_facts, dict):
                    print(f"[WARNING] Spatial facts not in dict format in {u['out_dir']}")
                    temp_spatial_facts = []
                    for s in _spatial_facts.values():
                        if isinstance(s, list):
                            temp_spatial_facts.extend(s)
                        else:
                            temp_spatial_facts.append(s)
                    _spatial_facts = temp_spatial_facts

                try:
                    captions.append(ViewCaption(
                        view_id=vm.view_id,
                        pano_size=u['pano_size'],
                        uv_meta={
                            "u_norm": u_norm, "v_norm": v_norm,
                            "fov_x_deg": vm.fov_x_deg,
                            "fov_y_deg": vm.fov_y_deg,
                            "fov_diag_deg": vm.fov_diag_deg,
                            "aspect_ratio": round(vm.aspect_ratio, 2),
                            "left_neighbor_view": vm.left_neighbor_view,
                            "right_neighbor_view": vm.right_neighbor_view,
                            "top_neighbor_view": vm.top_neighbor_view,
                            "bottom_neighbor_view": vm.bottom_neighbor_view,
                        },
                        caption=_caption,
                        objects=_objects,
                        spatial_facts=_spatial_facts,
                    ))
                except Exception as e:
                    print(f"[ERROR] parsing error: {e} for {u['out_dir']}")
                    captions.append(ViewCaption(
                        view_id=vm.view_id,
                        pano_size=u['pano_size'],
                        uv_meta={
                            "u_norm": u_norm, "v_norm": v_norm,
                            "fov_x_deg": vm.fov_x_deg,
                            "fov_y_deg": vm.fov_y_deg,
                            "fov_diag_deg": vm.fov_diag_deg,
                            "aspect_ratio": round(vm.aspect_ratio, 2),
                            "left_neighbor_view": vm.left_neighbor_view,
                            "right_neighbor_view": vm.right_neighbor_view,
                            "top_neighbor_view": vm.top_neighbor_view,
                            "bottom_neighbor_view": vm.bottom_neighbor_view,
                        },
                        caption="Empty caption - parsing error",
                        objects=[],
                        spatial_facts=[],
                    ))
            save_variable(captions, u['out_dir'], "captions")
            batch_captions[u['idx']] = captions

    # fill the cached captions
    for idx, item in enumerate(batch_item_0):
        if batch_captions[idx] is None:
            batch_captions[idx] = item

    # summarizing
    batch_view_paths = [[os.path.join(Path(batch_out_dir[idx]).parent, vm.view_id + ".jpg") for vm in item] for idx, item in enumerate(batch_item_0)]
    batch_result = agent.summarize(batch_captions, batch_view_paths, batch_pano_label)
    for idx, result in enumerate(batch_result):
        # format handler
        if not isinstance(result, dict):
            result = {"summary": result}
            print(f"[WARNING] Summary not in dict format in {batch_out_dir[idx]}")

        pano_summary = PanoSummary(
            pano_id=Path(batch_pano_path[idx]).stem, 
            pano_size=batch_pano_size[idx],
            pano_path=batch_pano_path[idx],
            summary=result.get("summary", ""),
            label=result.get("label", "") if "unknown" in batch_pano_label[idx].lower() or batch_pano_label[idx] is None else batch_pano_label[idx],
            outdoor=str(result.get("outdoor", "Not Given")),
        )
        save_variable(pano_summary, batch_out_dir[idx], "summary")

    # clean up the patch views
    remove_patch_views(
        [str(Path(batch_out_dir[idx]).parent) for idx in range(len(batch_out_dir))]
    )

def main(args):
    # load cfg
    cfg = load_config(path=args.config_path)
    data_dir = cfg["data_dir"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    cfg["agent_config"]["limit_mm_per_prompt"] = {"image": 13}

    print(cfg)

    # prepare all files
    df = pd.read_csv(cfg["annotation_path"])
    pano_files = df.itertuples(index=False, name=None)
    pano_files = [(os.path.join(data_dir, o[0].split("_")[0], o[0]), o[1]) for o in pano_files]

    print(f"Find {len(pano_files)} panoramas to process")

    # init agent
    agent = build_vllm_agent(cfg)

    # prepare the batch
    max_workers = min(8, os.cpu_count() or 4)
    batch_size = cfg["batch_size"]
    inflight_limit = batch_size * 2    
    prepared_buffer = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        pbar = tqdm(total=len(pano_files), desc="Processing panoramas")

        pending = set()
        i_submit = 0

        while i_submit < len(pano_files) and len(pending) < inflight_limit:
            fut = ex.submit(process_pano, pano_files[i_submit], cfg, agent.backend, args.overwrite)
            pending.add(fut)
            i_submit += 1
        
        while pending:
            for fut in as_completed(list(pending), timeout=None):
                pending.remove(fut)
                try:
                    returns = fut.result()
                    if returns:
                        prepared_buffer.append(returns)
                except Exception as e:
                    print(f"[ERROR] Error processing panorama: {e}")

                pbar.update(1)

                if len(prepared_buffer) >= batch_size:
                    infer_and_flush(prepared_buffer[:batch_size], agent)
                    prepared_buffer = prepared_buffer[batch_size:]

                while i_submit < len(pano_files) and len(pending) < inflight_limit:
                    fut_new = ex.submit(process_pano, pano_files[i_submit], cfg, agent.backend, args.overwrite)
                    pending.add(fut_new)
                    i_submit += 1

                break

        while prepared_buffer:
            chunk = prepared_buffer[:batch_size]
            infer_and_flush(chunk, agent)
            prepared_buffer = prepared_buffer[len(chunk):]

        pbar.close()
    agent.model.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./configs/batch_config.yaml")
    parser.add_argument("--overwrite", "-o", action="store_true")
    args = parser.parse_args()
    main(args)