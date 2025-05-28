import matplotlib.pyplot as plt
import torchvision

def show_images(dataloader, all_preds, all_targets, epochs, batches):
    class_names = ['day', 'night']
    #dataiter = iter(dataloader)
    #images, labels = next(dataiter)
    runs = 599
    # unnormalize for display
    def unnormalize(img):
        return img * 0.5 + 0.5

    # Create a copy to modify for color marking
    i = 0
    images_colored = [len(all_preds)][3]
    while (i < all_preds):
        print (f"LABELS item:" + str(all_targets[i]))
        print (f"PREDS item:" + str(all_preds[i]))
        if all_targets[i] == all_preds[i]:
            # Correct prediction → green tint
            images_colored[i][0] = 0.0  # no red
            images_colored[i][1] = 1.0  # green
            images_colored[i][2] = 0.0  # No blue
        else:
            # Incorrect prediction → red tint
            images_colored[i][0] = 1.0
            images_colored[i][1] = 0.0
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
    print('Predicted:', [class_names[p] for p in all_preds])


