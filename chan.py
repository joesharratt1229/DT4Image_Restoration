import h5py

def list_all_names(h5file, path='/'):
    with h5py.File(h5file, 'r') as f:
        return _recursively_list_names(f, path)

def _recursively_list_names(h5group, path):
    names = []
    for key in h5group[path]:
        item_path = f'{path}{key}/' if isinstance(h5group[path + key], h5py.Group) else path + key
        print(item_path)
        names.append(item_path)
        if isinstance(h5group[path + key], h5py.Group):
            names.extend(_recursively_list_names(h5group, item_path))
    return names

# Replace 'yourfile.h5' with the path to your HDF5 file
file_names = list_all_names('dataset/data/state_dir/data_1.h5')
