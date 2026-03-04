import h5py


def explore_afm_file(file_path):

    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name} shape={obj.shape}")
        else:
            print(f"GROUP: {name}")

    with h5py.File(file_path, "r") as f:
        print("Exploring AFM file structure\n")
        f.visititems(print_structure)