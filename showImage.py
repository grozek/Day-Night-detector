import matplotlib.pyplot as plt
import torchvision
import torch
import math

def show_images(dataloader, all_preds, all_targets, epochs, batches, image_size=32):
    class_names = ['day', 'night']
    #dataiter = iter(dataloader)
    #images, labels = next(dataiter)
    runs = 599
    # unnormalize for display
    def unnormalize(img):
        return img * 0.5 + 0.5
    num_images = len(all_preds)
    images_colored = torch.zeros(num_images, 3, image_size, image_size)
    # Create a copy to modify for color marking
    i = 0
    correct = 0
    wrong = 0
    for i in range(num_images):
        if all_preds[i] == all_targets[i]:
            # Green for correct
            images_colored[i, 1] = 1.0  # Green channel full
            correct = correct + 1
        else:
            # Red for incorrect
            images_colored[i, 0] = 1.0  # Red channel full
            wrong = wrong + 1
    print(f"STATS: wrong: " + str(wrong) + " correct: " + str(correct) + " average correct: " + str(correct/(correct+wrong)))


    # Display the color-coded batch
    grid = torchvision.utils.make_grid(images_colored, nrow=int(math.sqrt(num_images)))
    plt.figure(figsize=(10, 5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.title("Green = Correct, Red = Incorrect")
    #print('Predicted:', [int(p) for p in all_preds])
    #print('Targets:  ', [int(t) for t in all_targets])
    plt.show()

    # Print actual labels