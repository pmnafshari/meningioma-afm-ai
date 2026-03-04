import h5py


class AFMReader:

    def __init__(self, file_path):
        self.file_path = file_path

    def open_file(self):
        return h5py.File(self.file_path, "r")

    def list_keys(self):
        with h5py.File(self.file_path, "r") as f:
            return list(f.keys())

    def read_dataset(self, key):
        with h5py.File(self.file_path, "r") as f:
            data = f[key][:]
        return data

    def explore_structure(self):
        with h5py.File(self.file_path, "r") as f:
            def print_structure(name, obj):
                print(name)
            f.visititems(print_structure)