import torch
from cl_framework.dataset_loader import DatasetLoader  # DatasetLoader dosyanın ismi neyse onu yaz

def main():
    dataset_name = "amb"  # "segerstolpe" da olabilir
    loader = DatasetLoader(dataset_name).load()

    X = loader.X
    y = loader.y

    save_path = loader.bench_path
    X_file = save_path / "X.pt"
    y_file = save_path / "y.pt"

    print(f"Saving tensors to:\n  {X_file}\n  {y_file}")
    torch.save(X, X_file)
    torch.save(y, y_file)
    print("Tensors saved successfully.")
    
if __name__ == "__main__":
    main()    