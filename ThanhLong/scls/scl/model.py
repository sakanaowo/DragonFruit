from pathlib import Path

from scls.engine.model import Model
from scls.models import scl
from scls.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
from scls.utils import ROOT, yaml_load


class SCL(Model):
    """SCL (You Only Look Once) object detection model."""

    def __init__(self, model="SCL11n.pt", task=None, verbose=False):
        """Initialize SCL model, switching to SCLWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if SCLWorld PyTorch model
            new_instance = SCLWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default SCL initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": scl.classify.ClassificationTrainer,
                "validator": scl.classify.ClassificationValidator,
                "predictor": scl.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": scl.detect.DetectionTrainer,
                "validator": scl.detect.DetectionValidator,
                "predictor": scl.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": scl.segment.SegmentationTrainer,
                "validator": scl.segment.SegmentationValidator,
                "predictor": scl.segment.SegmentationPredictor,
            },
        }


class SCLWorld(Model):
    """SCL-World object detection model."""

    def __init__(self, model="SCLv8s-world.pt", verbose=False) -> None:
        """
        Initialize SCL-World model with a pre-trained model file.

        Loads a SCL-World model for object detection. If no custom class names are provided, it assigns default
        COCO class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": scl.detect.DetectionValidator,
                "predictor": scl.detect.DetectionPredictor,
                "trainer": scl.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes