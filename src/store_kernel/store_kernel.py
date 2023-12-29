import numpy as np
import h5py
from typing import Tuple, Dict, Any  # for clear function signatures

class Hdf5Kernel:
    def __init__(self, filename: str):

        self._filename = filename
        self._kernel_dims = None  # Dimensions are set when writing or reading for the first time


    @property
    def kernel_dims(self):
        """
        getter function that returns the kernel dimensions if thy are set and otherwise tries to read them from the file.
        """
        if self._kernel_dims is None:
            try: 
                with h5py.File(self._filename, "r") as hdf:
                    self._kernel_dims = np.array(hdf["kernel_dims"][:])

            except FileNotFoundError:
                raise FileNotFoundError(f"File {self._filename} not found. Unable to retrieve 'kernel_dims'.")
            
            except KeyError:
                raise KeyError(f"'kernel_dims' dataset not found in {self._filename}.")
            
            except Exception as e:
                raise RuntimeError(f"An error occurred while accessing {self._filename}: {e}")

        return self._kernel_dims
       

    def store_kernel_data(self, kernel_object: np.ndarray) -> None:
        """
        Stores an array of kernel objects in an HDF5 file. Each group correspond to one grid_point

        Args:
            kernel_object: The kernel object to be stored. Expected to have a 'shape' attribute,
                        a 'flatten' method, and a 'get_params' method.
        """
        

        # Check if the array is a scalar (zero-dimensional)
        if np.asarray(kernel_object).ndim == 0:
            # Reshape scalar to a 1 dimensional array
            kernel_object = np.array([kernel_object])

        else:
            # Convert kernel_object to a NumPy array (no effect if it's already an array)
            kernel_object = np.asarray(kernel_object)

        self._kernel_dims = np.array(kernel_object.shape)
        # flatten kernel object if it is not already 1D
        kernel_flat = kernel_object.ravel()

        with h5py.File(self._filename, "w") as hdf:
            hdf.create_dataset(
                "kernel_dims", data=self._kernel_dims
            )  # store dimensions of kernel_dims as attribute associated with file.

            for i, k in enumerate(kernel_flat):
                # Create a unique group for each grid point
                group_name = f"grid_point.idx_{i}"
                grid_point_group = hdf.create_group(group_name)

                params = k.get_params()  # get all parameters of the kernel class
                param_keys = []  # keep track of all variables that are stored as attributes
                for key, value in params.items():
                    grid_point_group.attrs[key] = value
                    param_keys.append(key)

                # store all kernel-related quantities except for the kernel matrix "kernel". Don't store parameters which are already stored as attributes above.
                attrs = vars(k)

                for key, value in attrs.items():
                    if key != "kernel" and key not in param_keys:
                        grid_point_group.create_dataset(key, data=value)



    def read_kernel_element(self, idx, isFlatIndex = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads kernel data from an HDF5 file and returns parameters and data for each element.

        Args:
            idx: (tuple or it): Index pointing to a specific point group. Tuple for multidimensional or int for flat arrays.
            isFlatIndex (bool): Flat must be set to true if multidimensional array should be accessed by specifing flat index
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing dictionaries (for parameters) and DataFrames (for data correspnding to the parameters),
                respectively. Each element corresponds to kernel data at a specific index.
        """
        
        with h5py.File(self._filename, "r") as hdf: 
          
            idx_flat = self._validate_and_flatten_index(idx, isFlatIndex=isFlatIndex)#convert to 1D index after validation

            group_name = f"grid_point.idx_{idx_flat}"
            grid_point_group = hdf[group_name]#access the group 
            params, data_df = self._access_kernel_element(grid_point_group)#read the data for the given grid point group

            return params, data_df
        

    def read_to_array(self):
        """
        Reads data from a given file and returns array for different quantities.

        Args:
        - None

        Returns:
        - errors (np.ndarray): Array with same dimensionality as original point grid holding all values for eps
        - m_vals (np.ndarray): Array with same dimensionality as original point grid holding all values for m
        - n_vals (np.ndarray): Array with same dimensionality as original point grid holding all values for n
        - h_vals (np.ndarray): Array with same dimensionality as original point grid holding all values for h 
        - ID_ranks (np.ndarray): Array with same dimensionality as original point grid holding all values for the ID rank
        """

        with h5py.File(self._filename, "r") as hdf:
       
            kernel_dims_flat = np.prod(self._kernel_dims)

            errors = np.empty((kernel_dims_flat,))
            m_vals = np.empty((kernel_dims_flat,))
            n_vals = np.empty((kernel_dims_flat,))
            h_vals = np.empty((kernel_dims_flat,))
            ID_ranks = np.empty((kernel_dims_flat,))

            for idx in range (kernel_dims_flat):
                #read parameters and data from file
                params, data = self.read_kernel_element(idx = idx, isFlatIndex = True)

                ID_ranks[idx] = data["ID_rank"] 

                errors[idx] = params["eps"]     
                m_vals[idx] = params["m"] 
                n_vals[idx] = params["n"] 
                h_vals[idx] = params["h"] 

        
        # reshape arrays to original shape when returning
        return (
            errors.reshape(self._kernel_dims),
            m_vals.reshape(self._kernel_dims),
            n_vals.reshape(self._kernel_dims),
            h_vals.reshape(self._kernel_dims),
            ID_ranks.reshape(self._kernel_dims),
        )

    def _validate_and_flatten_index(self, idx, isFlatIndex: bool):
        """
        Validates an index against kernel dimensions and converts it to a flattened index.

        Parameters:
        idx (tuple or int): Index to be validated, tuple for multidimensional arrays or int for flat arrays.
        isFlatIndex (bool): Flat must be set to true if multidimensional array should be accessed by specifing flat index

        Returns:
        int: The flattened one-dimensional index.

        Raises:
        IndexError: If 'idx' does not match the kernel's dimensions or is out of range.
        """
        if isFlatIndex:
            kernel_dims_flat = np.prod(self.kernel_dims)

            if not isinstance(idx, int):
                raise IndexError (f"When isFlatindex = True, index must be an integer, got {idx}.")
            if not 0 <= idx < kernel_dims_flat:
                raise IndexError(f"Flat index {idx} out of range. Valid range: [0, {kernel_dims_flat}).")

            return idx
        
        else:
            if not isinstance(idx, tuple):
                raise TypeError(f"When isFlatIndex = False, index must be an a tuple of integers. Got {idx}.")
            
            if len(idx) != len(self.kernel_dims):
                raise IndexError(f"Index dimensions {len(idx)} do not match kernel dimensions {len(self.kernel_dims)}.")

            if any(dim < 0 or dim >= size for dim, size in zip(idx, self.kernel_dims)):
                raise IndexError(f"Multi-dimensional index {idx} out of range for kernel dimensions {self.kernel_dims}.")

            idx_flat = np.ravel_multi_index(idx, self.kernel_dims)
            return idx_flat

           



    def _access_kernel_element(self, grid_point_group) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reads a kernel object from the given HDF5 group.

        Args:
            grid_point_group: HDF5 group object corresponding to a specific kernel data.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: A tuple containing two dictionaries.
                The first dictionary contains all parameters associated with the kernel.
                The second one contains the data for that kernel.
        """
        params = {attr: grid_point_group.attrs[attr] for attr in grid_point_group.attrs}
        data = {
            key: grid_point_group[key][()]
            for key in grid_point_group.keys()
            if key not in params
        }

        return params, data