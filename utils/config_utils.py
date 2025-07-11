"""
Utilities for generating Triton Inference Server configuration files.
"""
import os
from pathlib import Path
from typing import List


def generate_triton_config(
    model_name: str,
    model_repository_path: str,
    gpu_ids: List[int],
    instance_count_per_gpu: int = 1,
) -> bool:
    """
    Generates a new 'config.pbtxt' for a Triton model by adding an
    instance_group section to a template file.

    Args:
        model_name: The name of the model.
        model_repository_path: The host path to the model repository.
        gpu_ids: A list of GPU device IDs to create instances on.
        instance_count_per_gpu: The number of instances to create on each GPU.

    Returns:
        True if the config was generated successfully, False otherwise.
    """
    model_dir = Path(model_repository_path) / model_name
    template_path = model_dir / "config.pbtxt.template"
    config_path = model_dir / "config.pbtxt"

    if not template_path.is_file():
        print(f"Error: Template file not found at {template_path}")
        return False

    # Read the template content
    template_content = template_path.read_text()

    # Generate the instance_group string
    instance_group_str = "instance_group [\n"
    for gpu_id in gpu_ids:
        instance_group_str += (
            f"  {{\n"
            f"    count: {instance_count_per_gpu}\n"
            f"    kind: KIND_GPU\n"
            f"    gpus: [ {gpu_id} ]\n"
            f"  }},\n"
        )
    # Remove the last comma
    if instance_group_str.endswith(",\n"):
        instance_group_str = instance_group_str[:-2] + "\n"
    instance_group_str += "]\n"

    # Combine template with the new instance group
    final_config = f"{template_content}\n\n{instance_group_str}"

    # Write the new config file
    config_path.write_text(final_config)
    print(f"Successfully generated config for model '{model_name}' at {config_path}")
    return True


def remove_triton_config(model_name: str, model_repository_path: str):
    """
    Removes the generated 'config.pbtxt' file for a model.

    Args:
        model_name: The name of the model.
        model_repository_path: The host path to the model repository.
    """
    config_path = Path(model_repository_path) / model_name / "config.pbtxt"
    if config_path.is_file():
        try:
            config_path.unlink()
            print(f"Removed generated config file: {config_path}")
        except OSError as e:
            print(f"Error removing config file {config_path}: {e}") 