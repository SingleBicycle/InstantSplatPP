# Prior Models Integration

This document describes the integration of additional prior models (VGGT, LSM, and SpatialTracker v2) into the InstantSplatPP codebase.

## Overview

The codebase now supports multiple prior models for geometry initialization and test pose estimation. The original MASt3R model is still the default, but you can now choose from:

- **MASt3R** (default): The original model used in InstantSplatPP
- **VGGT**: VGGT prior model
- **LSM**: LSM prior model  
- **SpatialTracker v2**: SpatialTracker v2 prior model

## Files Modified

### 1. `init_geo.py`
- Added model type selection parameter (`--model_type`)
- Created `load_prior_model()` factory function
- Added imports for new prior models with graceful fallback
- Added default checkpoint paths for each model type
- Updated main function to use the selected model

### 2. `init_test_pose.py`
- Applied the same changes as `init_geo.py` for consistency
- Added model type selection and factory function

### 3. `example_usage.py` (new)
- Example script demonstrating how to use different prior models
- Shows command-line usage for each model type

## Usage

### Basic Usage

```bash
# Using MASt3R (default)
python init_geo.py --source_path ./assets/sora/Art --model_path ./output/mast3r

# Using VGGT
python init_geo.py --source_path ./assets/sora/Art --model_path ./output/vggt --model_type vggt

# Using LSM
python init_geo.py --source_path ./assets/sora/Art --model_path ./output/lsm --model_type lsm

# Using SpatialTracker v2
python init_geo.py --source_path ./assets/sora/Art --model_path ./output/spatial_tracker_v2 --model_type spatial_tracker_v2
```

### Custom Checkpoint Paths

```bash
# Specify custom checkpoint path
python init_geo.py --source_path ./assets/sora/Art --model_path ./output/custom --model_type vggt --ckpt_path /path/to/custom/checkpoint.pth
```

### Test Pose Initialization

The same functionality is available in `init_test_pose.py`:

```bash
python init_test_pose.py --source_path ./assets/sora/Art --model_path ./output/test --model_type vggt
```

## Default Checkpoint Paths

The following default checkpoint paths are used for each model type:

- **mast3r**: `./mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
- **vggt**: `./vggt/checkpoints/VGGT_model.pth`
- **lsm**: `./lsm/checkpoints/LSM_model.pth`
- **spatial_tracker_v2**: `./spatial_tracker_v2/checkpoints/SpatialTrackerV2_model.pth`

## Model Integration Requirements

To use the new prior models, you need to:

1. **Install the model dependencies**: Each model may require specific dependencies that need to be installed separately.

2. **Place model checkpoints**: Put the model checkpoint files in the expected directories or specify custom paths.

3. **Implement model classes**: The code expects the following model classes to be available:
   - `vggt.model.VGGTModel`
   - `lsm.model.LSMModel`
   - `spatial_tracker_v2.model.SpatialTrackerV2Model`

   Each model class should have a `from_pretrained()` class method that returns a model instance.

## Error Handling

The code includes graceful error handling:

- If a model is not available (import fails), a warning is printed and the model is set to `None`
- If you try to use an unavailable model, an `ImportError` is raised with a helpful message
- If you specify an unknown model type, a `ValueError` is raised with supported options

## Backward Compatibility

The changes are fully backward compatible:

- Default behavior remains unchanged (uses MASt3R)
- All existing command-line arguments work as before
- No changes to the core functionality, only model loading

## Example Integration

Here's how you would integrate a new prior model:

1. **Add the import** (with error handling):
```python
try:
    from your_model.model import YourModel
except ImportError:
    YourModel = None
    print("Warning: YourModel not available")
```

2. **Add to the factory function**:
```python
elif model_type.lower() == 'your_model':
    if YourModel is None:
        raise ImportError("YourModel is not available. Please install the required dependencies.")
    return YourModel.from_pretrained(ckpt_path).to(device)
```

3. **Add to argument choices**:
```python
parser.add_argument('--model_type', type=str, default='mast3r', 
    choices=['mast3r', 'vggt', 'lsm', 'spatial_tracker_v2', 'your_model'],
    help='Type of prior model to use')
```

4. **Add default checkpoint path**:
```python
default_checkpoints = {
    # ... existing models ...
    'your_model': './your_model/checkpoints/YourModel_checkpoint.pth'
}
```

## Testing

To test the new functionality:

1. Run the example script: `python example_usage.py`
2. Test with each model type using the provided commands
3. Verify that the correct model is loaded by checking the console output

## Notes

- The model factory function assumes all models follow the same interface (have a `from_pretrained()` method)
- If your model has a different interface, you may need to modify the factory function accordingly
- Make sure to place the model checkpoint files in the correct directories or specify custom paths
