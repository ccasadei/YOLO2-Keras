import json


class Config:

    def __init__(self, config_path):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epochs']
        self.base_lr = config['train']['base_lr']
        self.patience = config['train']['patience']
        self.do_freeze_layers = config['train']['do_freeze_layers']
        self.train_val_split = config['train']['train_val_split']
        self.augmentation = config['train']['augmentation']

        self.pretrained_weights_path = config['path']['pretrained_weights']
        self.base_weights_path = config['path']['base_weights']
        self.trained_weights_path = config['path']['trained_weights']
        self.chkpnt_weights_path = config['path']['chkpnt_weights']
        self.backend_weights = config['path']['backend_weights']
        self.images_path = config['path']['images']
        self.annotations_path = config['path']['annotations']
        self.test_images_path = config['path']['test_images']
        self.test_result_path = config['path']['test_result']
        self.log_path = config['path']['log']

        self.input_size = config['model']['input_size']
        self.classes = config['model']['classes']

