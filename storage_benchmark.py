import numpy as np
import h5py
import os
import time
from scipy.sparse import csr_matrix

from pathlib import Path

from hexrd import imageseries
from hexrd import matrixutil
import numba
import progiter
import argparse
import zarr
from numcodecs import Blosc

COMPRESSION_ARGS = {"compression": "gzip", "compression_opts": 9, "shuffle": True}
COMPRESSION_ARGS = {"compression": "lzf", "shuffle": True}
COMPRESSION_ARGS = {
    "compression": "szip",
}
COMPRESSION_ARGS = {
    "compression": "gzip",
    "compression_opts": 6,
    "shuffle": True,
    "chunks": True,
}


filepath = Path("test-fe-loadonly-1_0001_EIG16M_CdTe_000000.h5").resolve()

ims = imageseries.open(filepath, format="eiger-stream-v1")


def test_method_list_arrays_method(
    filename, ims, threshold, compression_args, nframes=None
):
    with h5py.File(filename, "w") as h5f:
        print(f"# single array:")
        start = time.perf_counter()

        size = ims.shape[0] * ims.shape[1]
        array = np.empty((size, 3), dtype=ims.dtype)
        if nframes is None:
            nframes = len(ims)
        for i in range(nframes):
            print(i)
            im = ims[i]
            row_array = array[:, 0]
            col_array = array[:, 1]
            data_array = array[:, 2]
            count = matrixutil.extract_ijv(
                im, threshold, row_array, col_array, data_array
            )
            frame = array[:count, :]
            # write as we read them to avoid running out of memory
            h5f.create_dataset(f"data_{i}", data=frame, **compression_args)
        h5f["shape"] = ims[0].shape
        h5f["nframes"] = nframes

    end = time.perf_counter()
    file_size = os.path.getsize(filename)
    write_array_time = end - start
    print(f"\t file size {file_size}")
    print(f"\t time to write {write_array_time}")


def test_method_list_arrays_method_zarr(
    filename, ims, threshold, compression_args, nframes=None
):
    with zarr.ZipStore(filename, mode="w") as store:
        file = zarr.open_group(store=store)
        print(f"# single array:")
        start = time.perf_counter()
        size = ims.shape[0] * ims.shape[1]
        array = np.empty((size, 3), dtype=ims.dtype)
        if nframes is None:
            nframes = len(ims)
        for i in range(nframes):
            im = ims[i]
            row_array = array[:, 0]
            col_array = array[:, 1]
            data_array = array[:, 2]
            count = matrixutil.extract_ijv(
                im, threshold, row_array, col_array, data_array
            )
            frame = array[:count, :]
            # write as we read them to avoid running out of memory
            file.create_dataset(
                f"data_{i}",
                data=frame,
                shape=frame.shape,
                dtype=frame.dtype,
                compressor=Blosc(cname="zstd", clevel=2, shuffle=Blosc.SHUFFLE),
            )  # , **compression_args)
        file.create_dataset(f"shape", data=ims[0].shape, dtype=np.uint32)
        file.create_dataset(f"nframes", data=nframes, dtype=np.uint32)

    end = time.perf_counter()
    file_size = os.path.getsize(filename)
    write_array_time = end - start
    print(f"\t file size {file_size}")
    print(f"\t time to write {write_array_time}")


def validate_list_arrays_method(filename, baseline_ims, nframes=None):
    print("Validate list_arrays method")
    with h5py.File(filename, "r") as file:
        shape = file["shape"][()]
        if nframes is None:
            nframes = file["nframes"][()]
        print(shape)
        print(nframes)

        for i in range(nframes):
            im = baseline_ims[i]
            frame_data = file[f"data_{i}"]
            row = frame_data[:, 0]
            col = frame_data[:, 1]
            data = frame_data[:, 2]
            frame = csr_matrix(
                (data, (row, col)), shape=shape, dtype=im.dtype
            ).toarray()
            np.testing.assert_array_equal(frame, im)


def test_single_array_method(filename, ims, threshold, compression_args, nframes=None):
    start = time.perf_counter()
    frame_size = ims.shape[0] * ims.shape[1]
    fixed_size = 5 * frame_size  # todo this is too big
    shape = ims.shape
    if nframes is None:
        nframes = len(ims)
    prev = 0
    buffer = np.zeros((frame_size, 3), dtype=ims.dtype)
    # creating an array in memory will fail if data is too big or threshold too low, so we write
    # to the file while iterating the frames
    with h5py.File(filename, "w") as h5f:
        dset = h5f.create_dataset(
            f"data",
            (100 * frame_size, 3),
            maxshape=(None, 3),
            dtype=ims.dtype,
            **compression_args,
        )
        h5f["shape"] = shape
        h5f["nframes"] = nframes
        prev = 0
        frame_indices = np.empty((len(ims) + 1,), dtype=np.uint64)
        for i in progiter.ProgIter(range(nframes)):
            im = ims[i]
            frame_indices[i] = prev
            row_slice = buffer[:, 0]
            col_slice = buffer[:, 1]
            data_slice = buffer[:, 2]
            count = matrixutil.extract_ijv(
                im, threshold, row_slice, col_slice, data_slice
            )
            # we need to copy back because changes in views in datasets do not reflect back
            dset[prev : prev + count, 0] = row_slice[:count]
            dset[prev : prev + count, 1] = col_slice[:count]
            dset[prev : prev + count, 2] = data_slice[:count]
            prev += count
            # when the remaming size drops below a full frame expand
            if dset.shape[0] - prev < frame_size:
                print("expand")
                dset.resize(dset.shape[0] + fixed_size, axis=0)
        frame_indices[len(ims)] = prev
        h5f.create_dataset(f"frame_ids", data=frame_indices, **compression_args)
    end = time.perf_counter()
    file_size = os.path.getsize(filename)
    print(f"#Method two")
    print(f"time :\t{end-start}")
    print(f"size :\t{file_size}")


def validate_single_array_method(filename, baseline_ims, nframes=None):
    print("Validate single_array method")
    with h5py.File(filename, "r") as file:
        all_data = file["data"]
        frame_indices = file["frame_ids"]
        shape = file["shape"][()]
        if nframes is None:
            nframes = file["nframes"][()]
        print(nframes)
        print(shape)

        for i in range(nframes - 1):
            print("testing frame", i)
            im = baseline_ims[i]
            frame_data = all_data[frame_indices[i] : frame_indices[i + 1]]
            row = frame_data[:, 0]
            col = frame_data[:, 1]
            data = frame_data[:, 2]
            frame = csr_matrix(
                (data, (row, col)), shape=shape, dtype=im.dtype
            ).toarray()
            np.testing.assert_array_equal(frame, im)


parser = argparse.ArgumentParser()
parser.add_argument("--threshold", default=4)
parser.add_argument("--filename", default="single_array.h5")
parser.add_argument("--validate", default=False)
parser.add_argument("--gzip", default=6)
parser.add_argument("--method", default=2)
parser.add_argument("--format", default="hdf5")
args = parser.parse_args()

nframes = None
threshold = args.threshold
if args.validate:
    threshold = -1
    nframes = 20
filename = args.filename
format = args.format
COMPRESSION_ARGS["compression_opts"] = int(args.gzip)
if args.method == "2":
    test_single_array_method(filename, ims, threshold, COMPRESSION_ARGS, nframes)
    if args.validate:
        validate_single_array_method(filename, ims, nframes)
else:
    if format == "hdf5":
        test_method_list_arrays_method(
            filename, ims, threshold, COMPRESSION_ARGS, nframes=nframes
        )
    else:
        test_method_list_arrays_method_zarr(
            filename, ims, threshold, COMPRESSION_ARGS, nframes=nframes
        )
    if args.validate:
        validate_list_arrays_method(filename, baseline_ims=ims, nframes=nframes)
