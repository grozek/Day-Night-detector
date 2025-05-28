import matplotlib.pyplot as plt
import torchvision

def show_images(dataloader, predicted, batch_size=16):
    class_names = ['day', 'night']
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # unnormalize for display
    def unnormalize(img):
        return img * 0.5 + 0.5

    images = unnormalize(images)
    num_to_show = min(len(images), len(predicted))

    # Create a copy to modify for color marking
    images_colored = images.clone()
    i = 0

    while (i < num_to_show):
        if labels[i] == predicted[i]:
            # Correct prediction → Red tint
            images_colored[i][0] = 1.0  # Red channel full
            images_colored[i][1] = 0.0  # No green
            images_colored[i][2] = 0.0  # No blue
        else:
            # Incorrect prediction → Green tint
            images_colored[i][0] = 0.0
            images_colored[i][1] = 1.0
            images_colored[i][2] = 0.0
        i = i + 1

    # Display the color-coded batch
    grid = torchvision.utils.make_grid(images_colored, nrow=4)
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Red = Correct, Green = Incorrect")
    plt.show()

    # Print actual labels
    print('Labels:   ', [class_names[label.item()] for label in labels])
    print('Predicted:', [class_names[p.item()] for p in predicted])


