import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Load the model (downloads from HuggingFace automatically)
model = build_sam3_image_model()
processor = Sam3Processor(model)

# Load your image
image = Image.open("/scratch/cl927/sam-3d-objects/scripts_volume/images/brunch.jpg")
inference_state = processor.set_image(image)

# Prompt with text (e.g., "apple", "person", "chair")
output = processor.set_text_prompt(state=inference_state, prompt="plate")

# Get results
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"mask shape: {masks.shape}, box shape: {boxes.shape}, scores shape: {scores.shape}")

# Save mask as a png
mask_image = Image.fromarray(masks[0].squeeze().cpu().numpy().astype("uint8") * 255)
mask_image.save("/scratch/cl927/sam-3d-objects/scripts_volume/sam3_masks/mask_plate.png")