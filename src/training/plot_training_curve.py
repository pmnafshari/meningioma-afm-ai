import matplotlib.pyplot as plt


def plot_training_curve(losses, model_name):

    plt.figure()

    epochs = range(len(losses))

    plt.plot(epochs, losses)

    plt.xlabel("epoch")
    plt.ylabel("loss")

    plt.title(f"training curve {model_name}")

    plt.savefig(f"results/{model_name}_training_curve.png")

    plt.close()