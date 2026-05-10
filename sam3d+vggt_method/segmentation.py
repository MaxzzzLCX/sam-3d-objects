import os

import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

def preprocess_image(image_path):
    """
    Preprocess the image into dimensions 518*518 for SAM-3.
     - Resize the image to 518*518 while maintaining aspect ratio and padding with zeros if necessary.
    """
    image = Image.open(image_path).convert('RGB')
    # First, pad the image into a square
    max_dim = max(image.size)
    padded_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))  # Create a black square image
    padded_image.paste(image, ((max_dim - image.size[0]) // 2, (max_dim - image.size[1]) // 2))  # Paste the original image in the center

    # Resize and pad to 518x518
    image = padded_image.resize((518, 518), Image.BILINEAR)
    return image


def segment_image(
    image_path: str,
    prompts: list[str],
):
    """
    Segment the image using SAM-3 with given prompts.
    """

    # Clear GPU cache to prevent CUDNN errors
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load the model (downloads from HuggingFace automatically)
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # Load your image
    # image_path = "/scratch/cl927/sam-3d-objects/scripts_volume/real_dataset/images/egg_2.jpeg"
    image_name = image_path.split("/")[-1].split(".")[0]
    # image = Image.open(image_path)
    image = preprocess_image(image_path)

    # Save the preprocessed image for visualization
    preprocessed_image_folder = f"{os.path.dirname(os.path.dirname(image_path))}/resized_images"
    os.makedirs(preprocessed_image_folder, exist_ok=True)
    image.save(f"{preprocessed_image_folder}/{image_name}.png")

    inference_state = processor.set_image(image)

    # mask_object_plate = "bowl"
    # mask_object_food = "egg"

    mask_folder = f"{os.path.dirname(os.path.dirname(image_path))}/masks_{image_name}"
    # mask_folder = f"/scratch/cl927/sam-3d-objects/scripts_volume/real_dataset/masks_{image_name}"
    os.makedirs(mask_folder, exist_ok=True)

    # Prompt with text (e.g., "apple", "person", "chair")
    for idx, prompt in enumerate(prompts):

        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        # Get results and save plate masks
        masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
        print(f"mask shape: {masks.shape}, box shape: {boxes.shape}, scores shape: {scores.shape}")
        if masks.shape[0] >  0:
            mask_image = Image.fromarray(masks[0].squeeze().cpu().numpy().astype("uint8") * 255)
            mask_image.save(f"{mask_folder}/{idx}.png")
        else:
            print(f"No masks found for prompt '{prompt}' in image '{image_path}'.")
    
    # Clean up GPU memory after processing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # # Save mask for food
    # output_food = processor.set_text_prompt(state=inference_state, prompt=mask_object_food)

    # masks, boxes, scores = output_food["masks"], output_food["boxes"], output_food["scores"]
    # mask_image = Image.fromarray(masks[0].squeeze().cpu().numpy().astype("uint8") * 255)
    # mask_image.save(f"{mask_folder}/1.png")

def batch_segment_images(image_folder, prompts):
    """
    Batch segment images in a folder.
    """
    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".JPG") or f.endswith(".jpeg") or f.endswith(".png")])
    for image_path in image_paths:
        segment_image(image_path, prompts)


def main():
    
    image_folders = [
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate/images",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_bowl/images",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_plate/images",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_bowl/images",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_plate/images",
        # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_bowl/images",
    #     "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_plate/images",
    #     "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_bowl/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_bowl/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/avocado_plate/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_bowl/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_plate/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_bowl/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_plate/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_bowl/images",
        "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_plate/images"
    ]
    dataset_prompts = [
        # ["plate", "orange"],
        # ["bowl", "orange"],
        # ["plate", "mango"],
        # ["bowl", "mango"],
        # ["plate", "box on the plate"],
        # ["white bowl", "box in the bowl"],
        # ["plate", "gum bottle"],
        # ["bowl", "gum bottle"],
        ["bowl", "avocado"],
        ["plate", "avocado"],
        ["bowl", "egg"],
        ["plate", "egg"],
        ["bowl", "pepper"],
        ["plate", "pepper"],
        ["bowl", "potato"],
        ["plate", "potato"]
    ]

    for image_folder, prompts in zip(image_folders, dataset_prompts):
        print(f"Segmenting images in {image_folder} with prompts {prompts}...")
        batch_segment_images(
            image_folder=image_folder,
            prompts=prompts
        )

if __name__ == "__main__":
    main()