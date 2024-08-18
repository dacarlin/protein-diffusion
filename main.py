from dataloader import get_dataloader
from model import DenoisingDiffusionModel
from training import train
from evaluation import evaluate_model

def main():
    pdb_dir = 'data'  # Directory containing your PDB files without extensions
    batch_size = 2
    epochs = 100
    learning_rate = 1e-4

    # Prepare DataLoader
    dataloader = get_dataloader(pdb_dir, batch_size)

    # Initialize the model
    model = DenoisingDiffusionModel()

    # Train the model
    train(model, dataloader, epochs=epochs, lr=learning_rate)

    # Evaluate the model
    evaluate_model(model, dataloader)

if __name__ == "__main__":
    main()
