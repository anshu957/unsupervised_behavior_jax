# %%
import pandas as pd
import h5py
import numpy as np
import shutil
import os
from tqdm import tqdm

folder_path = "/flashscratch/chouda/VAME/datasets/jabs600_v2/jabs250/poses/"
dest_path = "/flashscratch/chouda/VAME/datasets/jabs600_v2/jabs250/poses_csv/"


# Create the destination directory if it doesn't exist
if not os.path.exists(dest_path):
    os.makedirs(dest_path)

# %%

for file in tqdm(os.listdir(folder_path)):

    h5_path = os.path.join(folder_path, file)

    try:
        h5_file = h5py.File(h5_path, "r")
    except:
        print("file ", file, "does not exist")
        continue

    numpy_arr = np.zeros(
        (
            h5_file["poseest"]["points"].shape[0],
            # (h5_file["poseest"]["points"].shape[1]) * 3, # for pose v2
            (h5_file["poseest"]["points"].shape[2]) * 3,  # for pose v6
        )
    )

    i = 0
    for j in range(h5_file["poseest"]["points"].shape[2]):
        numpy_arr[:, 3 * i] = h5_file["poseest"]["points"][
            :, 0, j, 1
        ]  # x is y in pose file
        numpy_arr[:, 3 * i + 1] = h5_file["poseest"]["points"][
            :, 0, j, 0
        ]  # y is x in pose file
        numpy_arr[:, 3 * i + 2] = h5_file["poseest"]["confidence"][:, 0, j]
        i += 1

    # Save the new dataframe to a csv file

    new_filename = os.path.join(dest_path, file.replace(".h5", ".csv"))

    df = pd.DataFrame(numpy_arr)
    # save the dataframe to a csv file
    df.to_csv(new_filename, index=False)

    # close the h5 file
    h5_file.close()


print("Done converting all files to csv")
