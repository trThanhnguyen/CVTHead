from dataset.video_data import get_deca_tform
from PIL import Image
from models.cvthead import CVTHead

import face_alignment
from skimage.transform import estimate_transform, warp, resize, rescale
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np

import os
import argparse

import tqdm
from sandbox.read_mesh import read_obj_to_tensor
from glob import glob
from pathlib import Path
import yaml
import pickle 
import logging 


def preprocess_image(img_pth, fa, device="cuda"):
    img = Image.open(img_pth)
    img = img.resize((256, 256))

    img_npy = np.array(img)     # (H=256,W=256,3), val:0-255
    img_npy = img_npy[:, :, :3]
    landmark = fa.get_landmarks(img_npy)[0]
    tform = get_deca_tform(landmark)    # (3,3)
          
    img_npy = img_npy / 255.       # (H,W,3), val: 0-1
    crop_image = warp(img_npy, tform.inverse, output_shape=(224, 224))   # (224, 224, 3), val:[0, 1]
 
    img_tensor = torch.from_numpy(img_npy).float()   # tensor, (H, W, 3), val: [0, 1]
    img_tensor = img_tensor.permute(2, 0, 1)    # (H, W, 3) --> (3, H, W)
    img_tensor = (img_tensor - 0.5) / 0.5      # (3,H,W), [-1, 1]

    crop_image = torch.tensor(np.asarray(crop_image)).float() # (224, 224, 3), val:0-1
    crop_image = crop_image.permute(2, 0, 1)             # (3, 224, 224), val:0-1
    tform = torch.tensor(np.asarray(tform)).float()     # (3,3)

    img_tensor = img_tensor.unsqueeze(0).to(device)    # (1,3,256,256)
    crop_image = crop_image.unsqueeze(0).to(device)    # (1,3,224,224)
    tform = tform.unsqueeze(0).to(device)              # (1,3,3)      
    return img_tensor, crop_image, tform


def driven_by_face(model, src_pth, drv_pth, out_pth, device, softmask=True):
    # face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa, device)
    drv_img, drv_img_crop, drv_tform = preprocess_image(drv_pth, fa, device)
    
    with torch.no_grad():
        outputs = model(src_img_crop, drv_img_crop, src_img, drv_img, src_tform, drv_tform, is_train=False, is_cross_id=True)
        predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
        predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

        # visualize
        predict_img = 0.5 * (predict_img + 1)
        predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
        predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
        if not softmask:
            predict_mask = (predict_mask > 0.6).float()     # (256,256,1), npy

        # apply mask
        predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
        predict_img = (predict_img * 255).astype(np.uint8)
        predict_img = Image.fromarray(predict_img)
        predict_img.save(out_pth)


# def audio_driven_flame(model, src_pth, out_pth, device, params_path, softmask=False):
    """
    Drive source image based on flame params from emote
    default frontal pose, change by passing pose
    """
#     # face landmark detector
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
#     src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa, device)
    
#     with open(params_path, 'rb') as f:
#         coeffs_dict = pickle.load(f)
#         assert type(coeffs_dict) == dict, f"Expect class:dict loaded from .pkl file, got {type(coeffs_dict)}"
    
    
def driven_by_mesh(model, src_pth, imgs_out_dir, expname, mesh_dir, device, softmask=False):

    os.makedirs(os.path.join(imgs_out_dir, expname))
    # face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa, device)
    
    # metadata for: (to avoid estimate deca again for every loop)
    # generate_from_mesh(src_img, src_tform, src_verts, src_codedict, src_mask, drv_verts, hair_deform=True, pose=None)
    src_verts, src_codedict, src_mask = model.get_mesh_metadata(src_img_crop, src_img)
    
    # read mesh from mesh folder
    mesh_paths = sorted(glob(mesh_dir + '/*.obj'))

    with torch.no_grad():
        for path in tqdm.tqdm(mesh_paths):
            filename = Path(path).stem
            verts = read_obj_to_tensor(path).to(device)
            outputs = model.generate_from_mesh(src_img, src_tform, src_verts, src_codedict, src_mask, verts)
            predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
            predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

            # visualize
            predict_img = 0.5 * (predict_img + 1)
            predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
            predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
            if not softmask:
                predict_mask = (predict_mask > 0.6) + 0.0     # (256,256,1), npy

            # apply mask
            predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
            predict_img = (predict_img * 255).astype(np.uint8)
            predict_img = Image.fromarray(predict_img)
            predict_img.save(os.path.join(imgs_out_dir, expname, filename + '.jpg'))            
            
            # frames.append(predict_img)
    # return frames

def driven_by_flame_coefs(model, src_pth, out_pth, device, softmask=False):
    # face landmark detector
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa, device)
    
    with torch.no_grad():
        pose = torch.zeros(1, 6).to(src_img.device) # (1, 6)    rotation (3) + jaw pose (3)

        # ######################### shape ###############################
        frames = []
        for i in range(10):
            shape = torch.zeros(1, 100).to(src_img.device) # (1, 100)
            shape[0, 0] =  2 * i / 10 
            outputs = model.flame_coef_generation(src_img_crop, src_img, src_tform, shape=shape, pose=pose)
            predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
            predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

            # visualize
            predict_img = 0.5 * (predict_img + 1)
            predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
            predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
            if not softmask:
                predict_mask = (predict_mask > 0.6) + 0.0     # (256,256,1), npy

            # apply mask
            predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
            predict_img = (predict_img * 255).astype(np.uint8)
            predict_img = Image.fromarray(predict_img)
            frames.append(predict_img)
            # predict_img.save("examples/shape_{}.png".format(i))
        frame_one = frames[0]
        
        os.makedirs(out_pth, exist_ok=True)
        out_name = os.path.join(out_pth, "shape.gif")
        frame_one.save(out_name, format="GIF", append_images=frames, save_all=True, duration=500, loop=0) 

        # ######################### exp ###############################
        frames = []
        for i in range(10):
            exp = torch.zeros(1, 100).to(src_img.device) # (1, 100)
            exp[0, 0] = 2 * i / 10 
            outputs = model.flame_coef_generation(src_img_crop, src_img, src_tform, exp=exp, pose=pose)
            predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
            predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

            # visualize
            predict_img = 0.5 * (predict_img + 1)
            predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
            predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
            if not softmask:
                predict_mask = (predict_mask > 0.6) + 0.0      # (256,256,1), npy

            # apply mask
            predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
            predict_img = (predict_img * 255).astype(np.uint8)
            predict_img = Image.fromarray(predict_img)
            frames.append(predict_img)
            # predict_img.save("examples/exp_{}.png".format(i))
        frame_one = frames[0]
        out_name = os.path.join(out_pth, "exp.gif")
        frame_one.save(out_name, format="GIF", append_images=frames, save_all=True, duration=500, loop=0) 

        # ######################### view ###############################
        frames = []
        for i in range(12):
            pose = torch.zeros(1, 6).to(src_img.device) # (1, 100)
            pose[0, 1] = - np.pi / 4 + i * np.pi / 24  
            outputs = model.flame_coef_generation(src_img_crop, src_img, src_tform, pose=pose)
            predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
            predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

            # visualize
            predict_img = 0.5 * (predict_img + 1)
            predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
            predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
            if not softmask:
                predict_mask = (predict_mask > 0.6) + 0.0      # (256,256,1), npy

            # apply mask
            predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
            predict_img = (predict_img * 255).astype(np.uint8)
            predict_img = Image.fromarray(predict_img)
            frames.append(predict_img)

        frame_one = frames[0]
        out_name = os.path.join(out_pth, "pose.gif")
        frame_one.save(out_name, format="GIF", append_images=frames, save_all=True, duration=500, loop=0) 


        # ######################### Jaw pose ###############################
        frames = []
        for i in range(12):
            pose = torch.zeros(1, 6).to(src_img.device) # (1, 100)
            pose[0, 3] = 0.5 * i / 12
            outputs = model.flame_coef_generation(src_img_crop, src_img, src_tform, pose=pose)
            predict_img = outputs["pred_drv_img"]   # (1,3,256,256), tensor, val:[-1,1]
            predict_mask = outputs["pred_drv_mask"] # (1,256,256), tensor, val:[0, 1], soft mask

            # visualize
            predict_img = 0.5 * (predict_img + 1)
            predict_img = predict_img[0].permute(1,2,0).cpu().numpy()   # (256,256,3), npy  
            predict_mask = predict_mask[0].permute(1,2,0).cpu().numpy()   # (256,256,1), npy
            if not softmask:
                predict_mask = (predict_mask > 0.6) + 0.0      # (256,256,1), npy

            # apply mask
            predict_img = predict_img * predict_mask + (1 - predict_mask)  # apply mask to predicted image, val:[0, 1], npy
            predict_img = (predict_img * 255).astype(np.uint8)
            predict_img = Image.fromarray(predict_img)
            frames.append(predict_img)

        frame_one = frames[0]
        out_name = os.path.join(out_pth, "jaw.gif")
        frame_one.save(out_name, format="GIF", append_images=frames, save_all=True, duration=500, loop=0) 

def main(args):
    device = "cuda"

    # >>>>>>>>>>>>>>>>> Model >>>>>>>>>>>>>>>>>
    model = CVTHead()                                        # cpu model 
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)        # gpu model

    # load pre-trained weights
    ckpt = torch.load(args.ckpt_pth, map_location="cpu")["model"]
    model.load_state_dict(ckpt, strict=False)
    print(f'-- Number of parameters (G): {sum(p.numel() for p in model.parameters())/1e6} M\n')

    if args.mesh_driven and args.mesh_dir is not None:
        driven_by_mesh(model, args.src_pth, args.save_dir, args.expname, args.mesh_dir, device, softmask=True) ###4 TODO: see what softmask does
    elif args.flame:
        driven_by_flame_coefs(model, args.src_pth, args.out_pth, device, softmask=True)
    else:
        driven_by_face(model, args.src_pth, args.drv_pth, args.out_pth, device, softmask=True)
    
    # if args.save_images:
    #     # Dimensions
    #     height, width, channels = frame[0].shape
    #     # Create video writer object
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change codec if needed (e.g., 'XVID')
    #     video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    #     for frame in frames:


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVTHead Inference')
    parser.add_argument('--src_pth', type=str, default="examples/1.png")
    parser.add_argument('--drv_pth', type=str, default="examples/2.png")
    parser.add_argument('--out_pth', type=str, default="examples/output.png")
    parser.add_argument('--save_dir', type=str, default="results")
    parser.add_argument('--expname', type=str, default="test")
    parser.add_argument('--ckpt_pth', type=str, default="data/cvthead.pt")
    parser.add_argument('--mesh_driven', type=bool, default="False")
    parser.add_argument('--mesh_dir', type=str, default="")
    parser.add_argument('--flame', action='store_true')
    # parser.add_argument('--save_images', action='store_true')
    # parser.add_argument('--save_video', action='store_true')
    args = parser.parse_args()

    main(args)
