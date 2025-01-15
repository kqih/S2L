import os
from cellpose import io, models, train, metrics
import cupy as np

class Trainer:
    def __init__(self, train_dir, test_dir=None, img_filter="_img", mask_filter="_masks", look_one_level_down=True):
        """
        Initializes the Trainer object and loads train-test data.

        :param train_dir: Path to the training directory.
        :param test_dir: Path to the testing directory (optional).
        :param img_filter: Suffix to filter image files.
        :param mask_filter: Suffix to filter mask files.
        :param look_one_level_down: Whether to search one level down for files.
        """
        self.train_dir = train_dir 

        self.test_dir = test_dir
        self.image_filter = img_filter
        self.mask_filter = mask_filter
        self.look_one_level_down = look_one_level_down

        # Set up logging for Cellpose
        io.logger_setup()

        # Validate and load data
        self.output = self.validate_and_load_data()

    def validate_and_load_data(self):
        try:
            print(f"Training Directory: {self.train_dir}")
            if self.test_dir:
                print(f"Testing Directory: {self.test_dir}")
            else:
                print("No testing directory provided.")

            # Load the data with the filters applied
            output = io.load_train_test_data(
                
                train_dir=self.train_dir,

                test_dir=self.test_dir if self.test_dir else None,
                image_filter=self.image_filter,
                mask_filter=self.mask_filter,
                look_one_level_down=self.look_one_level_down
            )

            # Unpack the output as in the example
            images, labels, image_names, test_images, test_labels, image_names_test = output
            return images, labels, image_names, test_images, test_labels, image_names_test

        except Exception as e:
            print(f"Unexpected error: {e}")
            raise ValueError(f"Unexpected error during validation: {e}")

    def train(self, model_name: str, model_type: str = "cyto3", channels: list = [1, 2], epochs: int = 100,
              learning_rate: float = 0.1, normalize: bool = True):
        """
        Trains a Cellpose model.

        :param model_name: Name for saving the trained model.
        :param model_type: Type of model to use (e.g., "cyto3").
        :param channels: List specifying input-output channel indices.
        :param epochs: Number of epochs for training.
        :param learning_rate: Learning rate for training.
        :param normalize: Whether to normalize images before training.
        :return: Path to the trained model and losses.
        """
        print("\nStarting training process...")
        print(f"Model: {model_type}, Channels: {channels}, Epochs: {epochs}, Learning Rate: {learning_rate}")

        # Extract data from the loaded dataset
        images, labels, _, test_images, test_labels, _ = self.output

        # Initialize the Cellpose model
        model = models.CellposeModel(model_type=model_type, gpu=True)

        # Train the model
        try:
            model_path, train_losses, test_losses = train.train_seg(
                model.net,
                train_data=images,
                train_labels=labels,
                channels=channels,
                
                normalize=normalize,
                test_data=test_images if self.test_dir else None,
                test_labels=test_labels if self.test_dir else None,
                weight_decay=1e-4,
                SGD=True,
                learning_rate=learning_rate,
                n_epochs=epochs,
                model_name=model_name
            )
            print(f"Training completed successfully! Model saved at: {model_path}")
            return model_path, train_losses, test_losses

        except Exception as e:
            raise RuntimeError(f"Error during training: {e}")
    def evaluate_precision(self):
        """
        Evaluates the precision of the trained model using the test dataset.

        :return: Precision score.
        """
        if not self.test_dir:
            raise ValueError("Test directory not provided. Cannot evaluate precision.")

        # Extract test images and labels
        _, _, _, test_images, test_labels, _ = self.output

        # Load the trained model
        model = models.CellposeModel(gpu=True)

        # Generate predictions for the test images
        predictions = []
        for img in test_images:
            masks, flows, styles, diams = model.eval(img, channels=[0, 0])  # Adjust channels as needed
            predictions.append(masks)

        # Flatten labels and predictions to compute precision
        y_true = np.concatenate([lbl.flatten() for lbl in test_labels])
        y_pred = np.concatenate([pred.flatten() for pred in predictions])

        # Calculate precision using Cellpose metrics
        precision = metrics.average_precision(y_true, y_pred)
        print(f"Precision: {precision:.4f}")
        return precision
