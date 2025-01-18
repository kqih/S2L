import torch
import numpy as np
import time
import threading
import os
from cellpose import io, models, transforms
import tqdm

class StopFlag:
    def __init__(self):
        self.stop = False
        self.model = models.Cellpose(gpu=torch.cuda.is_available(), model_type='cyto')  # Initialize Cellpose model

    def segment(self, directory, diameter, progress_callback=None):
        start_time = time.time()

        files = [filename.path for filename in os.scandir(directory) if filename.is_file()]
        total_files = len(files)

        if total_files == 0:
            if progress_callback:
                progress_callback(100)  # If no files, set progress to 100%
            print("No files to process.")
            return

        stop_flag = StopFlag()

        for idx, filename in enumerate(tqdm.tqdm(files)):
            if stop_flag.stop:
                break

            # Load the image and handle different image types
            img = io.imread(filename)

            if len(img.shape) == 2:  # Grayscale
                channels = [[0, 0]]
            elif len(img.shape) == 3:  # RGB or multi-channel
                if img.shape[2] == 3:  # RGB
                    channels = [[2, 1], [1, 2]]  # Example: Red-Green and Green-Red
                else:  # Other multi-channel images
                    channels = [[0, 0]]  # Default to no channels
            else:
                print(f"Unsupported image shape: {img.shape}, skipping...")
                continue

            img_smoothed = transforms.smooth_sharpen_img(img, smooth_radius=1, sharpen_radius=0)

            for chan in channels:
                masks, flows, styles, diams = self.model.eval(
                    img_smoothed, 
                    diameter=diameter, 
                    channels=chan, 
                    flow_threshold=0.3, 
                    cellprob_threshold=0
                )
                img_normalized = transforms.normalize99(img, lower=1, upper=99)

                def save_masks_with_timeout(img, masks, flows, filename):
                    stop_flag_local = StopFlag()

                    def save_masks_thread(img):
                        io.save_masks(img, masks, flows, filename, save_txt=False)

                    t = threading.Timer(30.0, lambda: setattr(stop_flag_local, "stop", True))
                    t.start()
                    thread = threading.Thread(target=save_masks_thread, args=(img,))
                    thread.start()
                    thread.join(timeout=30.0)
                    t.cancel()

                    if thread.is_alive():
                        print("Saving masks taking too long, skipping...")
                    else:
                        pass

                save_masks_with_timeout(img, masks, flows, filename)

            if progress_callback:
                progress = ((idx + 1) / total_files) * 100
                progress_callback(progress)

        if progress_callback:
            progress_callback(100)  # Ensure progress is set to 100% after completion

        print(f"Segmentation completed in {time.time() - start_time:.2f} seconds")
