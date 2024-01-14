import numpy as np
import h5py
from typing import Tuple, Dict, Any  # for clear function signatures
import os


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
        self, kernel_object, idx, isFlatIndex: bool = False
    ) -> None:
        """
        Appends a kernel to an HDF5 file. Creates a new group for the kernel object.

        Args:
        - kernel_object: The kernel object to be appended. Must have a 'get_params' method.
        - idx (tuple or int): Index pointing to a specific point group. Tuple for multidimensional or int for flat arrays.
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

            params = kernel_object.get_params()
            param_keys = set(params.keys())
            for key, value in params.items():
                grid_point_group.attrs[key] = value

            # Store all kernel-related quantities except for those already stored as attributes
            #and except for the spectral density which is a callable function
            for key, value in vars(kernel_object).items():
                if key not in ("kernel", "spec_dens") and key not in param_keys:
                    grid_point_group.create_dataset(key, data=value)

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
            raise ValueError("Invalid kernel_object: Must be a non-empty NumPy array.")

        # in case kernel_object is zero-dimensional, convert to 1D array with one element
        kernel_object = np.atleast_1d(kernel_object)

        if np.any(kernel_object.shape != self._kernel_dims):
            raise ValueError(
                f"Shape of kernel array {kernel_object.shape} does not match specified kernel dimensions {self._kernel_dims}."
            )

        kernel_flat = kernel_object.ravel()

        try:
            for i, k in enumerate(kernel_flat):
                self.append_kernel_element(kernel_object=k, idx=i, isFlatIndex=True)
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
        """

        with h5py.File(self._filename, "r") as hdf:
            idx_flat = self._validate_and_flatten_index(
                idx, isFlatIndex=isFlatIndex
            )  # convert to 1D index after validation

            group_name = f"grid_point.idx_{idx_flat}"

            if group_name in hdf:
                # Attempt to access the group directly
                grid_point_group = hdf[group_name]

                # If the group is found, read its data
                params, data = self._access_kernel_element(grid_point_group)
                return params, data

            else:
                # If the group does not exist in the file, return default values.
                params = {
                    "N_max": 0,
                    "beta": 0.0,
                    "delta_t": 0.0,
                    "eps": 0.0,
                    "h": 0.0,
                    "m": 0,
                    "n": 0,
                    "phi": 0.0,
                }
                data = {
                    "ID_rank": 0,
                    "coarse_grid": np.array([]),
                    "fine_grid": np.array([]),
                    "idx": np.array([]),
                    "k_values": np.array([]),
                    "nbr_sv_above_eps": 0,
                    "proj": np.array([]),
                    "singular_values": np.array([]),
                    "times": np.array([]),
                    "spec_dens_array_cmplx": np.array([])
                }
                return params, data

    def read_to_array(self):
        """
        Reads data from a given file and returns array for different quantities.

        Args:
        - None

        Returns:
        - Tuple[dict, np.ndarray]: Tuple containing a dictionary with data arrays, corresponding to different quantities at all parameter combinations, and np.ndarray which holds the kernel dimensions.
        """

        with h5py.File(self._filename, "r") as hdf:
            kernel_dims_flat = np.prod(self.kernel_dims)

            errors = np.empty((kernel_dims_flat,))
            m_vals = np.empty((kernel_dims_flat,))
            n_vals = np.empty((kernel_dims_flat,))
            h_vals = np.empty((kernel_dims_flat,))
            N_maxs = np.empty((kernel_dims_flat,))
            betas = np.empty((kernel_dims_flat,))
            ID_ranks = np.empty((kernel_dims_flat,))
            delta_t_vals = np.empty((kernel_dims_flat,))

            for idx in range(kernel_dims_flat):
                # read parameters and data from file
                params, data = self.read_kernel_element(idx=idx, isFlatIndex=True)

                ID_ranks[idx] = data["ID_rank"]

                errors[idx] = params["eps"]
                m_vals[idx] = params["m"]
                n_vals[idx] = params["n"]
                h_vals[idx] = params["h"]
                betas[idx] = params["beta"]
                N_maxs[idx] = params["N_max"]
                delta_t_vals[idx] = params["delta_t"]

        # reshape arrays to original shape when returning
        return (
            {
                "errors": errors.reshape(self._kernel_dims),
                "m_vals": m_vals.reshape(self._kernel_dims),
                "n_vals": n_vals.reshape(self._kernel_dims),
                "h_vals": h_vals.reshape(self._kernel_dims),
                "betas": betas.reshape(self._kernel_dims),
                "N_maxs": N_maxs.reshape(self._kernel_dims),
                "ID_ranks": ID_ranks.reshape(self._kernel_dims),
                "delta_t_vals": delta_t_vals.reshape(self._kernel_dims),
            },
            self.kernel_dims,
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
