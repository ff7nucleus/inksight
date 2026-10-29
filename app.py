import glob
import time

import gradio as gr
import torch
from torchvision import transforms

from model import InkNet, prepare

# this loads the 47 character labels in the same order the model learned them
with open("class_names.txt") as f:
    CLASSES = f.read().split()

# this rebuilds the network and loads the trained weights on cpu
model = InkNet(len(CLASSES))
model.load_state_dict(torch.load("inksight_cnn.pth", map_location="cpu"))
model.eval()

# this converts a 28x28 image into the normalized tensor format used in training
to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


# this reads one character and times the full prediction from raw input to answer
def predict(img):
    # this pulls the finished drawing out of the sketchpad's layered output
    if isinstance(img, dict):
        img = img.get("composite")
    if img is None:
        return {}, "Draw or upload a character first."

    # this starts the clock, prepares the image and runs the model
    start = time.perf_counter()
    small = prepare(img)
    if small is None:
        return {}, "No ink detected. Try a darker or larger character."
    with torch.inference_mode():
        probs = model(to_tensor(small).unsqueeze(0)).softmax(1)[0]
    elapsed = time.perf_counter() - start

    # this returns every class score for the label panel plus the timing
    return {c: float(p) for c, p in zip(CLASSES, probs)}, f"{elapsed:.4f} seconds"


# this is the page layout: input tabs on the left, results on the right
with gr.Blocks(title="InkSight") as demo:
    gr.Markdown(
        "# InkSight: Handwritten Character Recognition\n"
        "Draw or upload one handwritten letter or digit. A convolutional neural network "
        "trained on EMNIST Balanced (47 classes) reads it and reports how long the prediction took."
    )
    with gr.Row():
        with gr.Column():
            # this is the drawing tab with a thick black marker
            with gr.Tab("Draw"):
                pad = gr.Sketchpad(
                    type="pil",
                    canvas_size=(400, 400),
                    brush=gr.Brush(default_size=28, colors=["#000000"], color_mode="fixed"),
                    layers=False,
                    label="Draw one character",
                )
                draw_btn = gr.Button("Read drawing", variant="primary")

            # this is the upload tab with clickable sample images
            with gr.Tab("Upload"):
                upload = gr.Image(type="pil", label="Upload a character image")
                upload_btn = gr.Button("Read image", variant="primary")
                samples = sorted(glob.glob("examples/*.png"))
                if samples:
                    gr.Examples(samples, inputs=upload)

        # this is the results panel: top three guesses and the inference time
        with gr.Column():
            label = gr.Label(num_top_classes=3, label="Prediction")
            timer = gr.Textbox(label="Inference time")

    # this connects each button to the prediction function
    draw_btn.click(predict, pad, [label, timer])
    upload_btn.click(predict, upload, [label, timer])

# this starts the web app when the file is run directly
if __name__ == "__main__":
    demo.launch()
