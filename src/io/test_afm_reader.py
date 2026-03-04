from afm_reader import AFMReader


file_path = "data/raw_afm/sample.h5"

reader = AFMReader(file_path)

print("Datasets in file:")
print(reader.list_keys())

print("\nFull structure:")
reader.explore_structure()