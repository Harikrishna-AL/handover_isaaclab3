import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from .rewards import dual_quaternion_error 

# Class to implement sequence storage
class TensorQueue:
    def __init__(self, max_size, element_shape):

        # Initialize the queue tensor with the max number of elements and given shape
        self.queue = torch.zeros((max_size, *element_shape))
        self.max_size = max_size
        self.current_size = 0
    
    def enqueue(self, new_element):
        # Shift elements to the left if the queue is full
        if self.current_size >= self.max_size:
            self.queue[:-1] = self.queue[1:].clone()
            self.queue[-1] = new_element
        else:
            # Add to the current end if the queue is not full
            self.queue[self.current_size] = new_element
            self.current_size += 1
    
    def get_queue(self):
        # Return only the filled part of the queue if not yet full
        return self.queue[:self.current_size]
    


# Reward computation
@torch.jit.script
def compute_rewards(rew_scale_hand_obj: float,
                    rew_scale_obj_target: float,
                    ee_pose: torch.Tensor,
                    obj_pose: torch.Tensor,
                    prev_dist: torch.Tensor,
                    rew_change_thres: float,
                    target_pose: torch.Tensor,
                    device: str):
    '''
    In:
        - rew_dual_quaternion_error - float: weighting factor for dual quaternion reward.
        - ee_pose - torch.tensor(N, 7): end effector pose in translation(3) + rotation in quaternions(4).
        - obj_pose - torch.tensor(N, 7): object pose in translation(3) + rotation in quaternions(4).
        - rew_change_thres - float: position threshold that changes the reach reward to the manipulation one.
        - device - str: Device into which the environment is stored.

    Out:
        - dq_reward - torch.tensor(N): rewards for all environments.
    '''

    # Dual quaternion distance between GEN3 hand and object
    hand_obj_dist = dual_quaternion_error(ee_pose, obj_pose, device)

    # Check if translation module is below the threshold
    obj_reached = hand_obj_dist[:, 1] < rew_change_thres
    
    # Dual quaternion distance between object and target pose
    obj_target_dist = dual_quaternion_error(obj_pose, target_pose, device)
    
    # Obtains the distance
    dist = hand_obj_dist[:, 0] * (~obj_reached) + obj_target_dist[:, 0] * obj_reached

    # Obtains wether the agent is approaching or not
    mod = 2*(dist < prev_dist) - 1

    # Compute intermediate reward terms with scaling values and boolean flags
    rew_term1 = mod * rew_scale_hand_obj * torch.exp(-2*hand_obj_dist[:, 0]) * (~obj_reached)
    rew_term2 = mod * rew_scale_obj_target * torch.exp(-2*obj_target_dist[:, 0]) * obj_reached

    # Obtain final reward
    reward = rew_term1 + rew_term2
    
    return reward, dist


# Reward computation -- joint space env
@torch.jit.script
def compute_rewards_joint_space(rew_scale_hand_obj: float,
                                rew_scale_obj_target: float,
                                fingertips_pose: torch.Tensor,
                                obj_pose: torch.Tensor,
                                rew_change_thres: float, 
                                target_pose: torch.Tensor,
                                device: str):
    
    # Compute distance between ee_pose and obj_pose
    avg_distance_fingertips_obj = torch.mean(torch.norm(fingertips_pose[:, :, :3] - obj_pose[:, :3].unsqueeze(dim=1).repeat(1, 4, 1),
                                              p=2, dim=-1), dim=1)
    
    # Compute distance between obj_pose and target_pose
    dist_obj_target = torch.norm(obj_pose[:, :3] - target_pose[:, :3], p=2, dim=-1)

    # Check if GEN3 has reached the object
    obj_reached = avg_distance_fingertips_obj < rew_change_thres

    # Compute reward terms
    rew_term1 = rew_scale_hand_obj * torch.exp(-3*avg_distance_fingertips_obj) * (~obj_reached)
    rew_term2 = rew_scale_obj_target * torch.exp(-3*dist_obj_target) * obj_reached

    reward = rew_term1 + rew_term2

    return reward


# Function to scale actions from [-1, 1] to [lower, upper]
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower

# Function to scale actions from [lower, upper] to [-1, 1]
@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


# Function for image saving from IsaacLab --> Taken from source/standalone/demos/cameras.py
def save_images_grid(
    images: list[torch.Tensor],
    cmap: str | None = None,
    nrow: int = 1,
    subtitles: list[str] | None = None,
    title: str | None = None,
    filename: str | None = None,
):
    """Save images in a grid with optional subtitles and title.

    Args:
        images: A list of images to be plotted. Shape of each image should be (H, W, C).
        cmap: Colormap to be used for plotting. Defaults to None, in which case the default colormap is used.
        nrows: Number of rows in the grid. Defaults to 1.
        subtitles: A list of subtitles for each image. Defaults to None, in which case no subtitles are shown.
        title: Title of the grid. Defaults to None, in which case no title is shown.
        filename: Path to save the figure. Defaults to None, in which case the figure is not saved.
    """
    
    # Show images in a grid
    n_images = len(images)
    ncol = int(np.ceil(n_images / nrow))

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2)) # -> tiene que ser un numpy array
    axes = np.array(axes)

    # Axes(0.125,0.11;0.775x0.77)
    axes = axes.flatten()

    # Plot images
    for idx, (img, ax) in enumerate(zip(images, axes)):
        img = img.detach().cpu().numpy()
        ax.imshow(img, cmap=cmap)
        ax.axis("off")
        if subtitles:
            ax.set_title(subtitles[idx])

    # Remove extra axes if any
    for ax in axes[n_images:]:
        fig.delaxes(ax)

    # Set title
    if title:
        plt.suptitle(title)

    # Adjust layout to fit the title
    plt.tight_layout()

    # Save the figure
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)

    # Close the figure
    plt.close()