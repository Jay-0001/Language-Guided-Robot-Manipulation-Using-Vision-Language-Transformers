from groundingdino.datasets.coco_grounding import COCODataset

class SphereDataset(COCODataset):
    """
    Your synthetic sphere dataset.
    Inherits everything from COCODataset, which already supports:
      - captions
      - token spans
      - normalized cxcywh bboxes
      - grounding fields
    """
    def __init__(self, img_folder, ann_file, transforms=None):
        super().__init__(
            img_folder,
            ann_file,
            transforms=transforms,
            return_masks=False
        )
