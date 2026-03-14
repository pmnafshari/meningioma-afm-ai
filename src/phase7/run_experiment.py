from model_factory import get_model
from experiment_logger import ExperimentLogger


def main():

    print("phase 7 experiment started")

    # create logger
    logger = ExperimentLogger()

    # example experiment configuration
    model_name = "resnet18"
    learning_rate = 0.001
    batch_size = 16
    optimizer_name = "adam"

    # create model (example with 3 classes)
    model = get_model(model_name, num_classes=3)

    print("model created:", model_name)

    # dummy accuracy for now (later replaced with real training result)
    accuracy = 0.0

    # log experiment
    logger.log(
        model_name,
        learning_rate,
        batch_size,
        optimizer_name,
        accuracy
    )

    print("experiment logged")


if __name__ == "__main__":

    main()