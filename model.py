import torch.nn as nn
from PIL import Image, ImageOps


# this is the cnn that reads a 28x28 handwritten character
class InkNet(nn.Module):
    def __init__(self, num_classes=47):
        super().__init__()

        # this learns low-level stroke patterns (edges, curves, line endings)
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        # this combines strokes into larger character shapes (loops, crossings)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25),
        )

        # this turns the learned shapes into one score per character class
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    # this runs an image through all three stages in order
    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


# this converts any drawing or photo into the same 28x28 format the model trained on
def prepare(img):
    # this flattens transparent sketch backgrounds onto white and converts to grayscale
    img = img.convert("RGBA")
    canvas = Image.new("RGBA", img.size, "white")
    canvas.alpha_composite(img)
    gray = canvas.convert("L")

    # this flips dark ink on light paper into light ink on black, matching the training data
    if sum(gray.getdata()) / (gray.width * gray.height) > 127:
        gray = ImageOps.invert(gray)

    # this crops tightly around the ink and returns nothing if the canvas is blank
    box = gray.point(lambda p: 255 if p > 50 else 0).getbbox()
    if box is None:
        return None
    gray = gray.crop(box)

    # this pads the character to a square so its shape is not stretched
    side = max(gray.size)
    square = Image.new("L", (side, side), 0)
    square.paste(gray, ((side - gray.width) // 2, (side - gray.height) // 2))

    # this shrinks it to 24x24 and centers it with a 2-pixel border, like the dataset
    out = Image.new("L", (28, 28), 0)
    out.paste(square.resize((24, 24), Image.LANCZOS), (2, 2))
    return out
