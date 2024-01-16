import numpy as np
import h5py
from typing import Tuple, Dict, Any  # for clear function signatures
import os
from src.kernel_matrix.kernel_matrix import KernelMatrix
import re#used for sorting keys in hdf5 file


class Hdf5Kernel:
    def __init__(self, filename: str):
        self._filename = filename
        self._kernel_dims = (
            None  # Dimensions are set when writing or reading for the first time
        )

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
                raise FileNotFoundError(
                    f"File {self._filename} not found. Unable to retrieve 'kernel_dims'."
                )

            except KeyError:
                raise KeyError(f"'kernel_dims' dataset not found in {self._filename}.")

            except Exception as e:
                raise RuntimeError(
                    f"An error occurred while accessing {self._filename}: {e}"
                )

        return self._kernel_dims


    def keys(self):
        """
        Returns a sorted list of all keys in the file.
        """
        #Regular expression to sort keys based on numerical content of key names
        def numerical_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        try:
            with h5py.File(self._filename, "r") as hdf:    
                return sorted(list(hdf.keys()), key=numerical_sort_key)

        except FileNotFoundError:
            raise FileNotFoundError(
                f"File {self._filename} not found. Unable to retrieve keys."
            )
        
    def create_file(self, kernel_dims):
        """
        Creates a new hdf5 file with the filename self._filename and sets the kernel dimensions used to convert between multiindices and scalar indices.
        """

        self._set_kernel_dims(kernel_dims=kernel_dims)

         #Comment the following if this function should overwrite exisiting files
        if os.path.exists(self._filename):
            raise FileExistsError(f"File '{self._filename}' already exists. Cannot create a new file.")

        try:
            with h5py.File(self._filename, "w") as hdf:
                hdf.create_dataset("kernel_dims", data=self.kernel_dims)
        except Exception as e:
            raise RuntimeError(f"An error occurred while creating the file: {e}")

        return self

    def _set_kernel_dims(self, kernel_dims):
        """
        Sets the kernel dimensions after validating that they are provided as a 1D array.

        Args:
            kernel_dims: The kernel dimensions to set. Must be a 1D array-like structure.

        Raises:
            ValueError: If 'kernel_dims' is None or not a 1D array.
        """

        if kernel_dims is None:
            raise ValueError("Kernel dimensions cannot be None.")

        kernel_dims_array = np.asarray(kernel_dims)
        if kernel_dims_array.ndim != 1:
            raise ValueError(
                f"Kernel dimensions must be given as a 1D array, got {kernel_dims} with {kernel_dims_array.ndim} dimensions."
            )

        self._kernel_dims = kernel_dims_array

    def append_kernel_element(
        self, idx, kernel_object, kernel_object2 = None, dict_data = None, isFlatIndex: bool = False
    ) -> None:
        """
        Append content one or two kernels to an HDF5 file. Creates a new group for the kernel object.

        Args:
        - idx (tuple or int): Index pointing to a specific point group. Tuple for multidimensional or int for flat arrays.
        - kernel_object: The kernel object to be appended. Must have a 'get_params' method.
        - kernel_object2 (optional): A supplementary kernel object to be appended. Must have a 'get_params' method.
        - dict_data (optional): A dictionary containing additional data to be stored.
        - isFlatIndex (bool): Specifies whether index refers to flattened or multidimensional array.

        Raises:
        - FileNotFoundError: If the HDF5 file does not exist.
        - IndexError: If the provided index is not an integer.
        - RuntimeError: If a group with the same index already exists.
        """

        if not os.path.exists(self._filename):
            raise FileNotFoundError(
                f"Cannot append data: File {self._filename} does not exist."
            )

        with h5py.File(self._filename, "r+") as hdf:
            idx_flat = self._validate_and_flatten_index(
                idx, isFlatIndex=isFlatIndex
            )  # convert to 1D index after validation

            group_name = f"grid_point.idx_{idx_flat}"
            if group_name in hdf:
                raise RuntimeError(f"Group {group_name} already exists in the file.")

            grid_point_group = hdf.create_group(group_name)

            #get parameters of main kernel object
            params = kernel_object.get_params()
            #and store their respective keys in an array
            used_keys = [*params.keys()]

            #store parameters as attributes for the group
            for key, value in params.items():
                grid_point_group.attrs[key] = value


            #_______Store kernel data________:
            #store kernel data (except 'kernel' (which may be large) and 'spec_dens' which is a callable function)
            self._store_kernel_data(group=grid_point_group, kernel_object=kernel_object, used_keys=used_keys)


            #_______Store supplementary kernel data (if provided)________:
            if kernel_object2 is not None:       
                #check that kernel objects are compatible. 
                self._kernel_objects_compatible(kernel_main = kernel_object, kernel2 = kernel_object2)#Raises Error if not compatible.
                #store kernel data (except 'kernel' (which may be large) and 'spec_dens' which is a callable function)
                self._store_kernel_data(group = grid_point_group, kernel_object = kernel_object2, used_keys = used_keys)

            #______Strore additional data (if provided)________:
            # Store all additional data except for those already stored as attributes
            if dict_data is not None:
                #store additional data
                self._store_dict_data(grid_point_group, dict_data, used_keys)

            
    def _kernel_objects_compatible(self, kernel_main: KernelMatrix, kernel2: KernelMatrix) -> None:
            """
            Check if the two kernel objects share the same base attributes

            Args:
                kernel_main (KernelMatrix): The main kernel object.
                kernel2 (KernelMatrix): The second kernel object.

            Raises:
                ValueError: If the parameters are not equivalent for a specific key.

            Returns:
                None
            """
            for key, val in kernel_main.get_shared_attributes().items():
                #check that all attributes are equivalent except for the spectral density which is a callable
                if key != "spec_dens":
                    #check that the attributes are equivalent
                    if not np.allclose(val, getattr(kernel2, key)):
                        raise ValueError(f"Attributes for kernel objects are not equivalent for key {key}: {val, getattr(kernel2, key)}.")
                    
                else:#check that the spectral density is equivalent for a range of frequencies
                    test_freq_array = np.arange(-1000,1000,0.1)
                    if not np.allclose(val(test_freq_array), getattr(kernel2, key)(test_freq_array)):
                        raise ValueError(f"Callables for spectral density give different results on test grid: {val(test_freq_array), getattr(kernel2, key)(test_freq_array)}")
        

    def _store_kernel_data(self, group, kernel_object, used_keys):
        """
        Stores the data of a kernel object in the given HDF5 group.
        """
        #check that kernel_object is of type KernelMatrix (typically is it one of its derived classes)
        if not isinstance(kernel_object, KernelMatrix):
            raise TypeError(f"Supplementary kernel object must be of type KernelMatrix, got {type(kernel_object)}.")

        # Store additional attributes of the kernel object
        for key, value in vars(kernel_object).items():
            #Store all kernel-related quantities except for those already stored 
            #Exclude also the spectral density which is a callable function
            #and the kernel matrix which may be a large object
            if key not in used_keys and key not in ["kernel", "spec_dens"]:
                group.create_dataset(key, data=value)
                used_keys.append(key)#append key to list of used keys

    def _store_dict_data(self, group, dict_data, used_keys):
        """
        Store dictionary data in an HDF5 group.

        Args:
            group (h5py.Group): The HDF5 group to store the data in.
            dict_data (dict): The dictionary containing the data to be stored.
            used_keys (list): A list of keys that have already been stored.

        Raises:
            ValueError: If `dict_data` is not a dictionary.

        """
        # Check that dict_data is a dict
        if not isinstance(dict_data, dict):
            raise ValueError(f"Additional data must be provided as a dictionary, got {type(dict_data)}.")

        for key, value in dict_data.items():
            # Store all kernel-related quantities except for those already stored
            # Exclude also the spectral density which is a callable function
            # and the kernel matrix which may be a large object
            if key not in ("kernel", "spec_dens") and key not in used_keys:
                group.create_dataset(key, data=value)
                used_keys.append(key)



    def store_kernel_array(self, kernel_object: np.ndarray) -> None:
        """
        Stores an array of kernel objects in an HDF5 file, where each element of the array
        represents a kernel object corresponding to a grid point.

        Args:
            kernel_object (np.ndarray): Array of kernel objects to be stored.

        Raises:
            ValueError: If 'kernel_object' is None, not an array, or its shape does not match 'self._kernel_dims'.
        """
        if kernel_object is None or not isinstance(kernel_object, np.ndarray):
            raise ValueError("Invalid kernel_object.")

        # in case kernel_object is zero-dimensional, convert to 1D array with one element
        kernel_object = np.atleast_1d(kernel_object)

        if np.any(kernel_object.shape != self._kernel_dims):
            raise ValueError(
                f"Shape of kernel array {kernel_object.shape} does not match specified kernel dimensions {self._kernel_dims}."
            )

        kernel_flat = kernel_object.ravel()

        try:
            for i, k in enumerate(kernel_flat):
                self.append_kernel_element(idx = i, kernel_object=k, isFlatIndex=True)
        except Exception as e:
            raise RuntimeError(f"An error occurred while storing kernel data: {e}")

    def read_kernel_element(
        self, idx, isFlatIndex=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reads kernel data from an HDF5 file and returns parameters and data for each element.

        Args:
            idx: (tuple or int): Index pointing to a specific point group. Tuple for multidimensional or int for flat arrays.
            isFlatIndex (bool): Flat must be set to true if multidimensional array should be accessed by specifing flat index
        Returns:
            Tuple[np.ndarray, np.ndarray]: Two arrays containing dictionaries (for parameters) and DataFrames (for data correspnding to the parameters),
                respectively. Each element corresponds to kernel data at a specific index.

        Raises:
            FileNotFoundError: If the HDF5 file does not exist.
            KeyError: If the group with the given index does not exist.
        """

        with h5py.File(self._filename, "r") as hdf:
            
            # convert to 1D index after validation
            idx_flat = self._validate_and_flatten_index(idx, isFlatIndex=isFlatIndex)  

            group_name = f"grid_point.idx_{idx_flat}"

            # Attempt to access the group directly
            if group_name in hdf:
                try:
                    grid_point_group = hdf[group_name]
                    params, data = self._access_kernel_element(grid_point_group)
                    return params, data
                except Exception as e:
                    raise RuntimeError(
                        f"An error occurred while accessing group {group_name}: {e}"
                    )
            else:
                raise KeyError(f"Group {group_name} not found in file {self._filename}.")

    def read_scalars_to_array(self, keys: list) -> dict:
        """
        Reads data from a given file and returns a dictionary where each key corresponds to a list of data items.

        Args:
        - keys (list): List of keys for which data is to be read from the file.

        Returns:
        - Tuple[list,dict]: first element: List containing the kernel dimensions. 
                            second element: A dictionary where each key maps to a numpy array of scalar data items.
        """

        kernel_dims_flat = np.prod(self.kernel_dims)
 
        # Initialize dictionary to hold data to be read out
        data_dict = {}
        
        for idx in range(kernel_dims_flat):
            # Read parameters and data from file
            params, data = self.read_kernel_element(idx=idx, isFlatIndex=True)
            
            for key in keys:
                if key in params:
                    data_read = params[key]
                    if data_read.ndim > 0:
                        raise ValueError(f"Parameter {key} is not a scalar.")
                elif key in data:
                    data_read = data[key]
                    if data_read.ndim > 0:
                        raise ValueError(f"Data {key} is not a scalar.")
                else:
                    raise KeyError(f"Key {key} not found in file {self._filename}.")
                
                # Append data to list in dictionary under the corresponding key
                data_dict.setdefault(key, []).append(data_read)

        # Convert lists to arrays
        for key, value in data_dict.items():
            data_dict[key] = np.array(value).reshape(self.kernel_dims)

        return self.kernel_dims, data_dict
    

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

            if not isinstance(idx, (int, np.int_)):
                raise IndexError(
                    f"When isFlatindex = True, index must be an integer, got {idx}."
                )
            if not 0 <= idx < kernel_dims_flat:
                raise IndexError(
                    f"Flat index {idx} out of range. Valid range: [0, {kernel_dims_flat})."
                )

            return idx

        else:
            if not isinstance(idx, tuple):
                raise TypeError(
                    f"When isFlatIndex = False, index must be an a tuple of integers. Got {idx}."
                )

            if len(idx) != len(self.kernel_dims):
                raise IndexError(
                    f"Index dimensions {len(idx)} do not match kernel dimensions {len(self.kernel_dims)}."
                )

            if any(dim < 0 or dim >= size for dim, size in zip(idx, self.kernel_dims)):
                raise IndexError(
                    f"Multi-dimensional index {idx} out of range for kernel dimensions {self.kernel_dims}."
                )

            idx_flat = np.ravel_multi_index(idx, self.kernel_dims)
            return idx_flat

    def _access_kernel_element(
        self, grid_point_group
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
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
