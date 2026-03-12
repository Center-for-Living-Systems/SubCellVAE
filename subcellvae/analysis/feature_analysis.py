import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
import tifffile as tiff
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from PIL import Image
import joblib
from umap import UMAP
from sklearn.cluster import DBSCAN
import pandas as pd
import czifile
import tifffile
from scipy.ndimage import distance_transform_cdt


def dataloader_AE_VAE_latents(model, dataloader, device, latent_source="mu"):
    model.eval()
    latents = []
    images = []
    group_ids = []
    with torch.no_grad():
        for x, group_id,_ in dataloader:
            x = x.to(device)
            
            out = model(x)

            # AE: (recon, z)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                recon, z = out
                latent = z

            # VAE/BetaVAE: (xhat, mu, logvar, z)
            elif isinstance(out, (tuple, list)) and len(out) == 4:
                xhat, mu, logvar, z = out
                latent = mu if latent_source == "mu" else z

            latents.append(latent.detach().cpu().numpy())
            images.append(x.cpu())
            group_ids.append(group_id)
    latents = np.concatenate(latents, axis=0)
    images = torch.cat(images, dim=0)
    return latents, images, group_ids


def dataloader_model_latents(model, dataloader, device, latent_source="mu"):
    model.eval()
    latents = []
    images = []
    group_ids = []
    with torch.no_grad():
        for x, group_id,_ in dataloader:
            x = x.to(device)
            
            out = model(x)

            # AE: (recon, z)
            if isinstance(out, (tuple, list)) and len(out) == 2:
                recon, z = out
                latent = z

            # VAE/BetaVAE: (xhat, mu, logvar, z)
            elif isinstance(out, (tuple, list)) and len(out) == 4:
                xhat, mu, logvar, z = out
                latent = mu if latent_source == "mu" else z

            latents.append(latent.detach().cpu().numpy())
            images.append(x.cpu())
            group_ids.append(group_id)
    latents = np.concatenate(latents, axis=0)
    images = torch.cat(images, dim=0)
    return latents, images, group_ids


def kmeans_cluster(latents, num_clusters, result_dir,kmeans_model_name):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(latents)
    labels = kmeans.labels_

    # Save the model
    joblib.dump(kmeans, os.path.join(result_dir, kmeans_model_name+'.pkl'))
    return kmeans, labels


def DBSCAN_cluster(latents, eps, min_samples, result_dir, DBSCAN_model_name):

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(latents)
    
    # Get cluster labels
    labels = db.labels_

    # Save the model
    joblib.dump(db, os.path.join(result_dir, DBSCAN_model_name+'.pkl'))
    return db, labels


# Step 4: Visualize Clusters with t-SNE and Reconstructed Images

def UMAP_train(latents, result_dir, umap_model_name):

    umap = UMAP(n_components=2, random_state=42)
    latents_2d = umap.fit_transform(latents)

    # Save model
    joblib.dump(umap, os.path.join(result_dir, umap_model_name+'.pkl'))
    return latents_2d


def latent_to_umap(umap_model_path, latents):

    # load model
    umap = joblib.load(umap_model_path)
    umap_latents_2d = umap.transform(latents)

    return umap_latents_2d

def kmeans_latents(kmeans_model_path, latents):
    # load the model
    kmeans = joblib.load(kmeans_model_path)
    labels = kmeans.predict(latents)

    return labels 


def data_to_latents(model, dataloader, device):
    model.eval()
    latents = []
    images = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            print(x.shape)
            _, z = model(x)
            latents.append(z.cpu().numpy())
            images.append(x.cpu())
    latents = np.concatenate(latents, axis=0)
    images = torch.cat(images, dim=0)
    print(images.shape)

    return latents, images


def patch_csv_to_AE_latent(ctrl_y_str, csv_folder, csv_filename,
                           group_labels, model_min, model_max,newdata_min, newdata_max,ae, device):

    all_image_csv = pd.read_csv(os.path.join(csv_folder, csv_filename))

    patch_label_df = pd.DataFrame(columns=[
        'ctrl_y_str', 'crop_img_filename', 'group_labels',
        'UMAP_d0', 'UMAP_d1', 'DBSCAN_labels', 'kmeans_labels'
    ])

    unique_vals = all_image_csv["filename"].unique()

    latent_list = []   # will become N x 8

    for filename in unique_vals:
        this_file_csv = all_image_csv[all_image_csv["filename"] == filename]

        for index, row in this_file_csv.iterrows():
            patch_name = row['crop_img_filename']
            raw_patch = tiff.imread(
                os.path.join(row["movie_partitioned_data_dir"],
                             row["crop_img_filename"])
            )

            if(model_max == newdata_max and model_min == newdata_min):
                tensor_patch = patch_2_normed_tensor(raw_patch, device)
            else:
                tensor_patch = histmatch_patch_2_normed_tensor(raw_patch, model_min, model_max, newdata_min, newdata_max, device)
                

            # ---- run AE ----
            with torch.no_grad():
                recon_image, latent = ae(tensor_patch)

            # ---- flatten latent → 1D vector ----
            latent_vec = latent.reshape(-1).detach().cpu().numpy()

            # ---- enforce 8-dim latent ----
            if latent_vec.shape[0] != 8:
                raise ValueError(
                    f"Expected latent dimension 8, but got {latent_vec.shape[0]} "
                    f"for patch {patch_name}."
                )

            latent_list.append(latent_vec)

            # ---- patch metadata ----
            s = pd.Series(
                [ctrl_y_str, patch_name, group_labels, 0, 0, 0, 0],
                index=['ctrl_y_str', 'crop_img_filename', 'group_labels',
                       'UMAP_d0', 'UMAP_d1', 'DBSCAN_labels', 'kmeans_labels']
            )
            patch_label_df = pd.concat([patch_label_df, s.to_frame().T],
                                       ignore_index=True)

    # Convert list → (N, 8) numpy matrix
    latent_array = np.vstack(latent_list)

    return patch_label_df, latent_array


def patch_2_normed_tensor(raw_patch,device):
    normed_raw_patch = raw_patch.copy() * 240 
    normed_raw_patch[normed_raw_patch > 254] = 254
    normed_raw_patch = normed_raw_patch/255
    tensor_patch = torch.from_numpy(normed_raw_patch)
    tensor_patch = tensor_patch.unsqueeze(0).unsqueeze(0)
    tensor_patch = tensor_patch.to(device)
    return tensor_patch
    
def histmatch_patch_2_normed_tensor(raw_patch,model_min, model_max,newdata_min, newdata_max,device):
    normed_01_raw_patch = (raw_patch.copy()*65535 - newdata_min)/(newdata_max-newdata_min)
    matched_raw_patch = (normed_01_raw_patch.copy()*(model_max-model_min) + model_min)/65535
    normed_raw_patch = matched_raw_patch.copy() * 240 
    normed_raw_patch[normed_raw_patch > 254] = 254
    normed_raw_patch = normed_raw_patch/255
    tensor_patch = torch.from_numpy(normed_raw_patch)
    tensor_patch = tensor_patch.unsqueeze(0).unsqueeze(0)
    tensor_patch = tensor_patch.to(device)
    return tensor_patch
    

def add_features_to_latent(csv_folder,csv_filename,czifolder,ctrl_y_str,cell_mask_folder,front_mask_dir,latent_array_np,patch_label_df):

    pad_size = 64

    all_image_csv = pd.read_csv(os.path.join(csv_folder,csv_filename))

    unique_vals = all_image_csv["filename"].unique()
    index_count=-1

    for filename in unique_vals:
        this_file_csv = all_image_csv[all_image_csv["filename"]==filename]

        raw_full_image = czifile.imread(os.path.join(czifolder, filename)).squeeze()    
        cell_mask = tifffile.imread(os.path.join(cell_mask_folder, "cell_mask_"+filename+".tif")).squeeze().astype(float)
      
        if os.path.isfile(os.path.join(front_mask_dir,'frontmask-'+filename[:-4]+'.tif')):
            front_mask = tifffile.imread(os.path.join(front_mask_dir,'frontmask-'+filename[:-4]+'.tif')).squeeze().astype(float)                         
        else:
            front_mask = cell_mask

        distance_taxicab = distance_transform_cdt(cell_mask, metric="taxicab")
            
        for index, row in this_file_csv.iterrows():       
            index_count = index_count+1 
            x_corner1 = int(row["x_corner1"])
            x_corner3 = int(row["x_corner3"])
            y_corner1 = int(row["y_corner1"])
            y_corner3 = int(row["y_corner3"])
            patch_name = row['crop_img_filename']

            matches = patch_label_df.index[
                (patch_label_df['crop_img_filename'] == patch_name) &
                (patch_label_df['ctrl_y_str'] == ctrl_y_str)
            ]

            if len(matches) == 0:
                print("No matching row found:", patch_name)
            elif len(matches) > 1:
                print("Warning: multiple matches found:", patch_name)

            full_image_ind = matches[0]

            # full_image_ind = patch_label_df.index[patch_label_df['crop_img_filename'] == patch_name][0]

            vinculin_intensity = raw_full_image[0,y_corner1-pad_size:y_corner3-pad_size,x_corner1-pad_size:x_corner3-pad_size].mean()
            pax_intensity = raw_full_image[1,y_corner1-pad_size:y_corner3-pad_size,x_corner1-pad_size:x_corner3-pad_size].mean()
            actin_intensity = raw_full_image[-1,y_corner1-pad_size:y_corner3-pad_size,x_corner1-pad_size:x_corner3-pad_size].mean()

            dist_val = distance_taxicab[int((y_corner1+y_corner3)/2)-pad_size,int((x_corner1+x_corner3)/2)-pad_size]

            front_flag = front_mask[int((y_corner1+y_corner3)/2)-pad_size,int((x_corner1+x_corner3)/2)-pad_size]

            patch_label_df.loc[full_image_ind, 'vinculin_intensity'] = vinculin_intensity
            patch_label_df.loc[full_image_ind, 'pax_intensity'] = pax_intensity
            patch_label_df.loc[full_image_ind, 'actin_intensity'] = actin_intensity
            patch_label_df.loc[full_image_ind, 'dist_cell_edge'] = dist_val
            patch_label_df.loc[full_image_ind, 'front_flag'] = front_flag
            
            # ⭐ NEW: also save to all_image_csv
            all_image_csv.loc[index, 'dist_cell_edge'] = dist_val
            all_image_csv.loc[index, 'front_flag'] = front_flag       
  

    latent_array_np_plus6 = np.hstack([
        latent_array_np,
        np.zeros((latent_array_np.shape[0], 6))  # 4 new columns
    ])

    latent_array_np_plus6[:,8] = patch_label_df['pax_intensity']
    latent_array_np_plus6[:,9] = patch_label_df['vinculin_intensity']
    latent_array_np_plus6[:,10] = patch_label_df['actin_intensity']
    latent_array_np_plus6[:,11] = patch_label_df['dist_cell_edge']
    latent_array_np_plus6[:,12] = patch_label_df['front_flag']
    latent_array_np_plus6[:,13] = patch_label_df['group_labels']

    return latent_array_np_plus6, patch_label_df, all_image_csv
                
            
def image_padding(input_img, pad_size,value):
    output_img = np.zeros([input_img.shape[0]+pad_size*2,input_img.shape[1]+pad_size*2]) + value
    output_img[pad_size:input_img.shape[0]+pad_size,pad_size:input_img.shape[1]+pad_size] = input_img
    return output_img