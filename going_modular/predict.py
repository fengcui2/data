"""
Predicts an image using a saved model 
"""
import torch
import torchvision
import argparse

import model_builder

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for image path
parser.add_argument("--image",  
                     help="the path of the image file")

# Get an arg for image path
parser.add_argument("--model_path", 
                    default='models/05_going_modular_script_mode_tinyvgg_model.pth', 
                    type=str,
                    help="target model to use for prediction filepath")

# parse the arguments
args = parser.parse_args()

# set up class names
class_names = ["pizza", "steak", "sushi"]

# set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# set up the image path
IMAGE_PATH = args.image
print(f"[INFO] Predicting on {IMAGE_PATH}")

# function to load the model
def load_model(filepath = args.model_path):
    model = model_builder.TinyVGG(
        input_shape = 3,
        hidden_units = 128,
        output_shape = 3).to(device)


    print(f"[INFO] Loading in model from {filepath}")
    model.load_state_dict(torch.load(filepath))
    return model

def predict_on_image(image_path=IMAGE_PATH, file_path=args.model_path):
    model = load_model(file_path)

    # Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(IMAGE_PATH)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255. 

    # Transform if necessary
    transform = torchvision.transforms.Resize(size=(64,64))
    image = transform(target_image)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        image = image.to(device) 

        # Make a prediction on image with an extra dimension and send it to the target device
        pred_logit = model(image.unsqueeze(dim=0))

        # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        pred_prob = torch.softmax(pred_logit, dim=1)

        # Convert prediction probabilities -> prediction labels
        pred_label = torch.argmax(pred_prob, dim=1)
        pred_label_class = class_names[pred_label]
    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_prob.max():.3f}")

if __name__ == "__main__":
  predict_on_image()

