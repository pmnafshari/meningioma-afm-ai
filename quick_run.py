import os
import sys
import subprocess
from pathlib import Path


def run_command(cmd):
    print("\nRUNNING:", cmd)
    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print("ERROR while running:", cmd)
        sys.exit(1)


def check_python():

    print("Checking Python version")

    if sys.version_info < (3,9):
        print("Python 3.9 or higher required")
        sys.exit(1)

    print("Python OK")


def setup_environment():

    if not Path("venv").exists():

        print("Creating virtual environment")

        run_command("python3 -m venv venv")

    print("Activating environment")

    activate_script = "source venv/bin/activate"

    run_command(f"{activate_script} && pip install --upgrade pip")


def install_requirements():

    print("Installing required libraries")

    run_command(
        "source venv/bin/activate && pip install torch torchvision pandas matplotlib scikit-learn h5py"
    )


def ask_dataset_path():

    dataset_path = input(
        "\nEnter path to AFM dataset folder (example: /Users/user/AFM_DATA): "
    )

    dataset_path = dataset_path.strip()

    if not Path(dataset_path).exists():

        print("Dataset path not found")
        sys.exit(1)

    return dataset_path


def prepare_project_folders():

    print("Preparing project folders")

    Path("data").mkdir(exist_ok=True)

    Path("data/dataset/train").mkdir(parents=True, exist_ok=True)
    Path("data/dataset/val").mkdir(parents=True, exist_ok=True)
    Path("data/dataset/test").mkdir(parents=True, exist_ok=True)

    Path("Output").mkdir(exist_ok=True)


def run_dataset_pipeline(dataset_path):

    print("\nRunning dataset preprocessing")

    os.environ["AFM_DATASET_PATH"] = dataset_path

    run_command(
        "source venv/bin/activate && python src/preprocessing/test_extract_curves.py"
    )

    run_command(
        "source venv/bin/activate && python src/datasets/test_build_real_dataset.py"
    )


def run_training():

    print("\nRunning model experiments")

    run_command(
        "source venv/bin/activate && python -m src.phase7.run_experiment"
    )


def run_evaluation():

    print("\nRunning evaluation")

    run_command(
        "source venv/bin/activate && python src/evaluation/evaluate_model.py"
    )


def run_gradcam():

    print("\nGenerating GradCAM visualizations")

    run_command(
        "source venv/bin/activate && python src/explainability/generate_gradcam_examples.py"
    )


def generate_plots():

    print("\nGenerating comparison plots")

    run_command(
        "source venv/bin/activate && python src/analysis/plot_model_comparison.py"
    )


def save_results():

    print("\nSaving final results")

    run_command("cp experiments/results.csv Output/")

    if Path("results").exists():
        run_command("cp -r results/* Output/")


def write_readme():

    text = """

AFM Deep Learning Project Results

model_comparison_plot.png
Comparison between tested architectures.

accuracy_bar_chart.png
Validation accuracy of models.

confusion_matrix.csv
Model prediction performance.

gradcam_examples
Visual explanation of model decisions.

experiments_results.csv
Raw experiment results.

"""

    with open("Output/results_explanation.txt","w") as f:
        f.write(text)


def main():

    print("\nAFM Deep Learning Project Pipeline\n")

    check_python()

    setup_environment()

    install_requirements()

    dataset_path = ask_dataset_path()

    prepare_project_folders()

    run_dataset_pipeline(dataset_path)

    run_training()

    run_evaluation()

    run_gradcam()

    generate_plots()

    save_results()

    write_readme()

    print("\nPROJECT FINISHED SUCCESSFULLY")
    print("Results saved in Output/")


if __name__ == "__main__":
    main()