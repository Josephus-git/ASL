import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset 
import os 
from PIL import Image 
import albumentations as A
import numpy as np
from colorama import Fore 
from matplotlib import pyplot as plt 
from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
import sys 


class DETRData(Dataset): 
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, 'labels')
        self.images_path = os.path.join(self.path, 'images')
        self.label_files = os.listdir(self.labels_path) 
        self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
        self.train = train
        
        # Initialize logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()
        
        # Log dataset initialization
        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path
        }
        self.data_handler.log_dataset_stats(dataset_info)
        
        # Log transforms information
        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Scale to [0,1] (Float32)",
            "Convert to Tensor"
        ]
        self.data_handler.log_transform_info(transform_list)             

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        self.transform = A.Compose(
            [   
                A.Resize(500,500),
                *([A.RandomCrop(width=224, height=224, p=0.33)] if self.train else []), # Example random crop
                A.Resize(224,224),
                *([A.HorizontalFlip(p=0.5)] if self.train else []),
                A.ToFloat(max_value=255),
                A.ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )
        
        for attempt in range(max_attempts):
            try:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
                # Check if we still have bboxes after transformation
                if len(transformed['bboxes']) > 0:
                    return transformed
            except:
                continue
        
        # Fallback: If stochastic augmentations fail to preserve boxes, apply only 
        # basic resizing and formatting to ensure we return Tensors, not numpy arrays.
        fallback_transform = A.Compose([
            A.Resize(224, 224),
            A.ToFloat(max_value=255),
            A.ToTensorV2()
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return fallback_transform(image=image, bboxes=bboxes, class_labels=labels)

    def __len__(self): 
        return len(self.labels) 

    def __getitem__(self, idx): 
        label_path = os.path.join(self.labels_path, self.labels[idx]) 
        image_name = self.labels[idx].split('.')[0]
        image_path = os.path.join(self.images_path, f'{image_name}.jpg') 
        
        img = Image.open(image_path).convert('RGB')
        with open(label_path, 'r') as f: 
            annotations = f.readlines()
        class_labels = []
        bounding_boxes = []
        for annotation in annotations: 
            annotation = annotation.split('\n')[:-1][0].split(' ')
            class_labels.append(annotation[0]) 
            bounding_boxes.append(annotation[1:])
        class_labels = np.array(class_labels).astype(int) 

        # Define the remapping based on the user's description of current labels
        # This assumes the original labels in the .txt files were:
        # 0: 'before'
        # 1: 'book' (implied, as 0, 2, 3 are taken)
        # 2: 'computer'
        # 3: 'drink'
        
        # Get the target class names and their desired IDs from config.json
        target_class_names = get_classes() # e.g., ['book', 'drink', 'computer', 'before']
        target_class_id_map = {name: idx for idx, name in enumerate(target_class_names)}
        
        # Create the remapping dictionary: original_id_from_txt -> desired_id_from_config
        remap_dict = {
            0: target_class_id_map['before'],   # Original 0 ('before') -> Desired 3 ('before')
            1: target_class_id_map['book'],     # Original 1 ('book') -> Desired 0 ('book')
            2: target_class_id_map['computer'], # Original 2 ('computer') -> Desired 2 ('computer')
            3: target_class_id_map['drink']     # Original 3 ('drink') -> Desired 1 ('drink')
        }
        
        # Apply the remapping to the class labels
        class_labels = np.array([remap_dict[label] for label in class_labels])
        bounding_boxes = np.array(bounding_boxes).astype(float) 

        augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
        augmented_img_tensor = augmented['image']
        augmented_bounding_boxes = np.array(augmented['bboxes'])
        augmented_classes = augmented['class_labels']

        labels = torch.tensor(augmented_classes, dtype=torch.long)  
        boxes = torch.as_tensor(augmented_bounding_boxes, dtype=torch.float32).reshape(-1, 4)
        return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

if __name__ == '__main__':
    CLASSES = get_classes()
    num_classes = len(CLASSES)
    dataset = DETRData('data/test', train=False) 
    dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=False)

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET) 
    fig, ax = plt.subplots(2,2) 
    axs = ax.flatten()
    for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
        ax.imshow(img.permute(1,2,0))
        box_classes = annotations['labels'] 
        boxes = rescale_bboxes(annotations['boxes'], (224,224))
        for box_class, bbox in zip(box_classes, boxes): 
            if box_class < num_classes: 
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                print(xmin, ymin, xmax, ymax) 
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
                text = f'{CLASSES[box_class]}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

    fig.tight_layout() 
    plt.show()     