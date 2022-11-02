class NumberOfFileNotSame(Exception):
    def __init__(self):
        super().__init__("The number of image files and label files doesn't same.")


class ModelTypeError(Exception):
    def __init__(self):
        super().__init__("The type of model must be onnx or pt.")
