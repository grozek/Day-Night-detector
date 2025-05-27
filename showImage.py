import matplotlib.pyplot as plt
import torchvision


def show_images(dataloader, predicted):
    class_names=['day', 'night']
    dataiter = iter(dataloader)
    images, labels = next(dataiter)  # get first batch

    # unnormalize for display
    def unnormalize(img):
        img = img * 0.5 + 0.5  # reverse normalization: [-1,1] → [0,1]
        return img

    images = unnormalize(images)

    grid = torchvision.utils.make_grid(images, nrow=4)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))  # convert [C,H,W] → [H,W,C] for matplotlib
    plt.axis('off')
    for image_nr in range (0, len(images)-1):
        if labels[image_nr] == predicted[image_nr]:
            images[image_nr[0, :, :]] = 1.0  # Red
            images[image_nr[1, :, :]] = 0.0  # green
            images[image_nr[2, :, :]] = 0.0  # blue

        if labels[image_nr] != predicted[image_nr]:
            images[image_nr[0, :, :]] = 0.0  # Red
            images[image_nr[1, :, :]] = 1.0  # green
            images[image_nr[2, :, :]] = 0.0  # blue

    # print labels below the image
    print('Labels:', [class_names[label.item()] for label in labels])
    plt.show()

