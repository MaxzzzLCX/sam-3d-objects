#!/bin/bash

# Render normalized meshes from NutritionVerse-3D dataset
# Usage: ./run_render.sh [--test_single food_item_name] [--n_views N]

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
RENDER_SCRIPT="$SCRIPT_DIR/render_nutrition_meshes.py"
BLENDER_PATH="/scratch/cl927/blender-3.6.5-linux-x64/blender"

# Default parameters
DATASET_PATH="/scratch/cl927/nutritionverse-3d-new"
N_VIEWS=10
TEST_SINGLE=""
LIMIT="-3"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test_single)
            TEST_SINGLE="$2"
            shift 2
            ;;
        --n_views)
            N_VIEWS="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --dataset_path PATH    Path to nutritionverse dataset (default: $DATASET_PATH)"
            echo "  --n_views N           Number of views per mesh (default: $N_VIEWS)"
            echo "  --limit N             Process only first N food items (useful for testing)"
            echo "  --test_single NAME    Test on single food item folder"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting mesh rendering with Blender..."
echo "Dataset: $DATASET_PATH"
echo "Views per mesh: $N_VIEWS"
if [ ! -z "$LIMIT" ]; then
    echo "Processing limit: $LIMIT food items"
fi

if [ ! -z "$TEST_SINGLE" ]; then
    echo "Testing single item: $TEST_SINGLE"
    $BLENDER_PATH --background --python "$RENDER_SCRIPT" -- \
        --dataset_path "$DATASET_PATH" \
        --n_views "$N_VIEWS" \
        --test_single "$TEST_SINGLE"
else
    echo "Processing entire dataset..."
    if [ ! -z "$LIMIT" ]; then
        $BLENDER_PATH --background --python "$RENDER_SCRIPT" -- \
            --dataset_path "$DATASET_PATH" \
            --n_views "$N_VIEWS" \
            --limit "$LIMIT"
    else
        $BLENDER_PATH --background --python "$RENDER_SCRIPT" -- \
            --dataset_path "$DATASET_PATH" \
            --n_views "$N_VIEWS"
    fi
fi

echo "Rendering complete!"