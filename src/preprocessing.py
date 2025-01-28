import pandas as pd
import h5py
import numpy as np
import shutil
import os
from tqdm import tqdm


def h5_to_csv_poses(FOLDER_PATH, DEST_PATH):
    """
    Converts all h5 files in the FOLDER_PATH to csv files and moves them to the DEST_PATH.
    """

    if not os.path.exists(DEST_PATH):
        os.makedirs(DEST_PATH)

    for file in tqdm(os.listdir(FOLDER_PATH)):

        h5_path = os.path.join(FOLDER_PATH, file)

        try:
            h5_file = h5py.File(h5_path, "r")
        except:
            print(f"file {file} does not exist")
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

        new_filename = os.path.join(DEST_PATH, file.replace(".h5", ".csv"))

        df = pd.DataFrame(numpy_arr)
        # save the dataframe to a csv file
        df.to_csv(new_filename, index=False)

        # close the h5 file
        h5_file.close()
        print("Done converting all files to csv and moving them to the destination folder.")


def pose_kp_filter(SOURCE_PATH, DEST_PATH, INDICES_TO_REMOVE):
    """
    Filter the pose keypoints and save them to a new directory
    Args:
        SOURCE_PATH: pathlib.Path object
            The path to the directory containing the pose keypoints
        DEST_PATH: pathlib.Path object
            The path to the directory where the filtered pose keypoints will be saved
        INDICES_TO_REMOVE: list
            The indices of the keypoints to remove
            For 10 kp, remove the last 2 tail key points, index 10,11
            For 8kp, remove forepaws and tail key points, index 4,5,10,11
    """

    # Create the destination directory if it doesn't exist
    DEST_PATH.mkdir(exist_ok=True)

    for file in tqdm.tqdm(list(SOURCE_PATH.glob("*.csv"))):

        # Read csv file
        try:
            df = pd.read_csv(file)
        except FileNotFoundError:
            print(f"Error reading {file}")
            continue

        # Remove the specified indices
        cols_to_remove = (
            [3 * i for i in INDICES_TO_REMOVE]
            + [3 * i + 1 for i in INDICES_TO_REMOVE]
            + [3 * i + 2 for i in INDICES_TO_REMOVE]
        )

        # Make entries as strings
        cols_to_remove = sorted(cols_to_remove)
        cols_to_remove = [str(i) for i in cols_to_remove]

        df_new = df.drop(cols_to_remove, axis=1)

        ## assert that the retained coloumns are the expected ones
        assert df_new.shape[1] == 3 * (12 - len(INDICES_TO_REMOVE))
        # assert that they are the same columns as the ones we expect
        for i in range(df_new.shape[1] // 3):
            # confidence columns are the same

            # indices not removed in the original dataframe
            indices_not_removed = [i for i in range(12) if i not in INDICES_TO_REMOVE]

            # assert that confidence columns for filtered dataframe are the same as the original dataframe
            assert df.iloc[:, 3 * indices_not_removed[i] + 2].equals(
                df_new.iloc[:, 3 * i + 2]
            )

        # Save the new dataframe to a csv file
        new_filename = DEST_PATH / file.name
        df_new.to_csv(new_filename, index=False)

    print("Done filtering all files to csv")
