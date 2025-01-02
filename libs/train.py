from cellpose import io, models, train

io.logger_setup()

class Trainer:
    def __init__(self, train_dir, test_dir):
        self.output = io.load_train_test_data(train_dir, test_dir, image_filter="_img", 
                                              mask_filter="_masks", look_one_level_down=False)

    def train(self, model_name: str, model: str = "cyto3", channels: list = [1, 2], epochs: int = 100, 
              learning_rate: float = 0.1, normalize: bool = True):
        images, labels, image_names, test_images, test_labels, image_names_test = self.output
        model_instance = models.CellposeModel(model_type=model)

        model_path, train_losses, test_losses = train.train_seg(
            model_instance.net,
            train_data=images,
            train_labels=labels,
            channels=channels,
            normalize=normalize,
            test_data=test_images,
            test_labels=test_labels,
            weight_decay=1e-4,
            SGD=True,
            learning_rate=learning_rate,
            n_epochs=epochs,
            model_name=model_name  # Using dynamic model_name passed from the function
        )

        # Returning the model path and the training/test losses
        return model_path, train_losses, test_losses
