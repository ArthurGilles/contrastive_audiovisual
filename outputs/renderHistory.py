import json
import matplotlib.pyplot as plt
from pathlib import Path


def load_history(file_path):
    """
    Loads the training history from a JSON file.
    Args:
        file_path (str): Path to the history JSON file.

    Returns:
        dict: Training history data or None if file not found or invalid.
    """
    try:
        with open(file_path, "r") as f:
            history = json.load(f)
        return history
    except FileNotFoundError:
        print(f"Error : The file {file_path} does not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error : The file {file_path} is not a valid JSON file.")
        return None


def extract_data(history_data, key_path):
    """
    Extracts epochs and values from the training history data based on the provided key path.
    Args:
        history_data (dict): History data loaded from JSON.
        key_path (list): List of keys to navigate through the history data.

    Returns:
        tuple: (epochs, values) or (None, None) if data is not found.
    """
    try:
        data = history_data
        for key in key_path:
            data = data[key]

        if not data:
            return None, None

        epochs = [point[0] for point in data]
        values = [point[1] for point in data]
        return epochs, values
    except (KeyError, IndexError):
        return None, None


def plot_training_history(history_file="history.json"):
    # All the comments and text in this function must be in English
    """
    Plots the training history of the model from a JSON file.
    Args:
        history_file (str): Path to the history JSON file.
    """
    # Load data
    history = load_history(history_file)
    if history is None:
        return

    # Set up the plot style and figure
    plt.style.use(
        "seaborn-v0_8" if "seaborn-v0_8" in plt.style.available else "default"
    )
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Model performance during training", fontsize=16, fontweight="bold")

    # First plot: Training Loss
    epochs_loss, loss_values = extract_data(history, ["training", "average_loss"])
    if epochs_loss is not None:
        axes[0, 0].plot(
            epochs_loss, loss_values, "b-", linewidth=2, marker="o", markersize=4
        )
        axes[0, 0].set_title("Training loss", fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Average Loss")
        axes[0, 0].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(
            0.5,
            0.5,
            "No loss data",
            ha="center",
            va="center",
            transform=axes[0, 0].transAxes,
        )
        axes[0, 0].set_title("Training loss (no data)", fontweight="bold")

    # Second plot: Audio Accuracy
    epochs_audio_train, audio_acc_train = extract_data(
        history, ["training", "audio_accuracy"]
    )
    epochs_audio_test, audio_acc_test = extract_data(
        history, ["test", "audio_accuracy"]
    )

    if epochs_audio_train is not None or epochs_audio_test is not None:
        if epochs_audio_train is not None:
            axes[0, 1].plot(
                epochs_audio_train,
                audio_acc_train,
                "g-",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Training",
            )
        if epochs_audio_test is not None:
            axes[0, 1].plot(
                epochs_audio_test,
                audio_acc_test,
                "r-",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Test",
            )

        axes[0, 1].set_title("Audio accuracy", fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No audio data",
            ha="center",
            va="center",
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Audio accuracy (no data)", fontweight="bold")

    # Third plot: Visual Accuracy
    epochs_visual_train, visual_acc_train = extract_data(
        history, ["training", "visual_accuracy"]
    )
    epochs_visual_test, visual_acc_test = extract_data(
        history, ["test", "visual_accuracy"]
    )

    if epochs_visual_train is not None or epochs_visual_test is not None:
        if epochs_visual_train is not None:
            axes[1, 0].plot(
                epochs_visual_train,
                visual_acc_train,
                "g-",
                linewidth=2,
                marker="o",
                markersize=4,
                label="Training",
            )
        if epochs_visual_test is not None:
            axes[1, 0].plot(
                epochs_visual_test,
                visual_acc_test,
                "r-",
                linewidth=2,
                marker="s",
                markersize=4,
                label="Test",
            )

        axes[1, 0].set_title("Visual accuracy", fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy (%)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(
            0.5,
            0.5,
            "No visual data",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )
        axes[1, 0].set_title("Visual accuracy (no data)", fontweight="bold")

    # Fourth plot: Comparison of Test Accuracies
    axes[1, 1].set_title("Comparison of test accuracy", fontweight="bold")
    if epochs_audio_test is not None and epochs_visual_test is not None:
        axes[1, 1].plot(
            epochs_audio_test,
            audio_acc_test,
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Audio",
        )
        axes[1, 1].plot(
            epochs_visual_test,
            visual_acc_test,
            "orange",
            linewidth=2,
            marker="s",
            markersize=4,
            label="Visual",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Accuracy (%)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(
            0.5,
            0.5,
            "No test data",
            ha="center",
            va="center",
            transform=axes[1, 1].transAxes,
        )

    # Adjust layout
    plt.tight_layout()

    plt.show()


def print_summary(history_file="history.json"):
    """
    Prints a summary of the model's performance based on the training history.

    Args:
        history_file (str): Path to the history JSON file.
    """
    history = load_history(history_file)
    if history is None:
        return

    print("\n" + "=" * 50)
    print("Summary of performance during training")
    print("=" * 50)

    # Summary of training loss
    epochs_loss, loss_values = extract_data(history, ["training", "average_loss"])
    if epochs_loss is not None and loss_values:
        print(f"Training Loss:")
        print(f"  - Start loss: {epochs_loss[0]}, Loss: {loss_values[0]:.4f}")
        print(f"  - Final epoch: {epochs_loss[-1]}, Loss: {loss_values[-1]:.4f}")
        print(
            f"  - Upgrade: {((loss_values[0] - loss_values[-1]) / loss_values[0] * 100):.1f}%"
        )

    # Summary of audio and visual accuracies
    for metric_name, path in [
        ("Audio", ["test", "audio_accuracy"]),
        ("Visual", ["test", "visual_accuracy"]),
    ]:
        epochs, values = extract_data(history, path)
        if epochs is not None and values:
            print(f"\nAccuracy {metric_name} (Test):")
            print(f"  - First measure : Epoch {epochs[0]}, {values[0]:.2f}%")
            print(f"  - Last measure: Epoch {epochs[-1]}, {values[-1]:.2f}%")
            print(
                f"  - Best performance: {max(values):.2f}% (Epoch {epochs[values.index(max(values))]})"
            )


if __name__ == "__main__":
    history_file = "history.json"

    # Check if the history file exists
    if not Path(history_file).exists():
        print(f"Error: The file {history_file} does not exist.")
    else:
        # Display summary
        print_summary(history_file)

        # Plot training history
        plot_training_history(history_file)
