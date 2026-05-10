source /scratch/cl927/miniconda3/etc/profile.d/conda.sh
conda activate vggt

# Define array of directories to process
SCENE_DIRS=(
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/testscene"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/orange_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/mango_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/box_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/gum_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/egg_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/pepper_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview/potato_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/egg_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/egg_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/orange_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/orange_plate"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/potato_bowl"
    # "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/potato_plate"
    "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/avocado_plate"
    "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/strawberry_bowl"
    "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/strawberry_plate"
    "/scratch/cl927/sam-3d-objects/scripts_volume/real_data_multiview_volume_vggt/orange_bowl_manual"
)

# Loop through each directory and run the script
for scene_dir in "${SCENE_DIRS[@]}"; do
    echo "Processing scene: $scene_dir"
    python /scratch/cl927/sam-3d-objects/sam3d+vggt_method/vggt_inference.py \
        --scene_dir "$scene_dir" \
        --conf_thres_value 0 \
        --mask \
        --construct_scenes
    echo "Completed: $scene_dir"
    echo "---"
done

echo "All scenes processed!" 