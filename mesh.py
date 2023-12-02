# %%
import os
import torch
from vmap import *
import open3d
vis_dict = {}
ckpt_path = '/home/wdebang/workspace/vMAP/logs/vMAP/room0/ckpt'
from cfg import Config
from tqdm import tqdm

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

config ='./configs/Replica/config_replica_room0_vMAP.json'
cfg = Config(config)
cam_info = cameraInfo(cfg)
intrinsic_open3d = open3d.camera.PinholeCameraIntrinsic(
    width=cfg.W,
    height=cfg.H,
    fx=cfg.fx,
    fy=cfg.fy,
    cx=cfg.cx,
    cy=cfg.cy)

# %%
for obj_id in os.listdir(ckpt_path):
    try:
        idx = int(obj_id)
    except ValueError:
        continue

    path = ckpt_path + "/" + obj_id
    if os.path.isdir(path):
        ckpt_name = path + '/' + os.listdir(path)[0]
        scene = sceneObject(cfg, idx)
        scene.load_checkpoints(ckpt_name)
        vis_dict[idx] = scene
        
# %%
frame_id = 1999
log_dir = '/home/wdebang/workspace/vMAP/logs/vMAP/room0'
with performance_measure("Exporting mesh"):
    for obj_id, obj_k in tqdm(vis_dict.items()):
        bound = obj_k.bbox3d
        if bound is None:
            print("get bound failed obj ", obj_id)
            continue
        if obj_id == 0:
            print("meshing bg")
            grid_dim = 256
        else:
            grid_dim = cfg.grid_dim
        adaptive_grid_dims = (np.clip(
            bound.extent // cfg.live_voxel_size + 1,
            2, 
            grid_dim
        ).astype(np.int32))
        # adaptive_grid_dims = (adaptive_grid_dims, adaptive_grid_dims, adaptive_grid_dims)
        print(f'Exporting mesh obj {obj_id}, grid_dim={adaptive_grid_dims}' + "~"*20)
        mesh = obj_k.trainer.meshing(bound, obj_k.obj_center, grid_dims=adaptive_grid_dims)
        if mesh is None:
            print("meshing failed obj ", obj_id)
            continue

        # save to dir
        obj_mesh_output = os.path.join(log_dir, "scene_mesh")
        os.makedirs(obj_mesh_output, exist_ok=True)
        mesh.export(os.path.join(obj_mesh_output, "frame_{}_obj{}.obj".format(frame_id, str(obj_id))))

# %%
vis_dict[0].bbox3d.extent
# %%
