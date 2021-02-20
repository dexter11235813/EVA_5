# GradCam source :- https://github.com/vickyliin/gradcam_plus_plus-pytorch
import torch
from numpy import ceil, floor, clip, sqrt, transpose, array
from torchvision import transforms
from gradcam.utils import visualize_cam
from gradcam import GradCAM
from show_images_from_batch import get_images_by_classification
import config
import matplotlib.pyplot as plt


def grad_cam(imgs, model, layer):
    images = []
    cam_config = dict(model_type="resnet", arch=model, layer_name=layer)
    if model.training:
        model.to(config.DEVICE).eval()
    for img in imgs:
        torch_img = transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor()]
        )(img).to(config.DEVICE)
        normed_img = transforms.Normalize(
            mean=[0.4914, 0.48216, 0.44653], std=[0.2023, 0.1994, 0.2010]
        )(torch_img)[None]
        cam = GradCAM.from_config(**cam_config)
        mask, _ = cam(normed_img)
        heatmap, _ = visualize_cam(mask, torch_img)

        images.extend([img.cpu(), heatmap, img.cpu() * 0.25 + heatmap * 0.75])

    return images


def modified_plot_grad_cam(
    number, model, test_loader, device, misclassified=True, save_path="./images"
):
    images, predicted, actual = get_images_by_classification(
        model, test_loader, device, misclassified
    )
    nrows = int(floor(sqrt(number)))
    ncols = int(ceil(sqrt(number)))
    if misclassified:
        save_path = f"{save_path}/misclassified_images_cam_{number}_images.png"
    else:
        save_path = f"{save_path}/correctly_classified_images_cam_{number}_images.png"

    cam_imgs = grad_cam.grad_cam(images[0:number], model, "layer4")
    cam_imgs = cam_imgs[2::3]
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            ax[i, j].set_title(
                f"Predicted: {config.CLASSES[predicted[index]]},\nActual : {config.CLASSES[actual[index]]}"
            )

            ax[i, j].axis("off")
            print(index)
            ax[i, j].imshow(clip(transpose(array(cam_imgs[index]), (1, 2, 0)), 0, 1))

    fig.savefig(save_path, bbox_inches="tight")
    print(f"plot saved at {save_path}")


def plot_grad_cam(
    number, model, test_loader, device, misclassified=True, save_path="./images"
):
    images, predicted, actual = get_images_by_classification(
        model, test_loader, device, misclassified
    )
    nrows = number
    ncols = 3
    if misclassified:
        save_path = f"{save_path}/misclassified_images_cam.png"
    else:
        save_path = f"{save_path}/correctly_classified_images_cam.png"

    cam_imgs = grad_cam(images[0:number], model, "layer4")
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 15))

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j

            if j == 0:
                ax[i, j].set_title(
                    f"Predicted: {config.CLASSES[predicted[i]]},\nActual : {config.CLASSES[actual[i]]}"
                )
            ax[i, j].axis("off")

            ax[i, j].imshow(clip(transpose(array(cam_imgs[index]), (1, 2, 0)), 0, 1))

    fig.savefig(save_path, bbox_inches="tight")
    print(f"plot saved at {save_path}")
