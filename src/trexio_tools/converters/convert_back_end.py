#!/usr/bin/env python3
"""
Convert a TREXIO file from one back end into another one.
"""


def generate_converter(json_filename: str) -> None:
    """
    Function that generates the converter based on the trex.json configuration file.
    """
    from os.path import join, dirname, abspath
    from json import load

    try:
        with open(json_filename, 'r') as f:
            config = load(f)
    except FileNotFoundError as e:
        raise Exception(f'File {json_filename} is not found.') from e

    fileDir     = dirname(abspath(__file__))
    output_file = join(fileDir, 'converter_generated.py')

    data_sparse      = []
    data_other       = []
    data_determinant = []
    for group,attr in config.items():
        for data,specs in attr.items():
            name = f'{group}_{data}'
            if 'sparse' in specs[0]:
                data_sparse.append(name)
            elif 'determinant' in group:
                data_determinant.append(name)
            else:
                data_other.append(name)

    with open(output_file, 'w') as f_out:
        f_out.write('import trexio \n')
        f_out.write('def data_handler(trexio_file_from, trexio_file_to) -> None : \n')
        f_out.write('  buffer_size = 20000 \n')

        # Process the normal data first
        for attr in data_other:
            if 'package_version' in attr:
                continue
            block = f'\n\
  if trexio.has_{attr}(trexio_file_from):        \n\
    data = trexio.read_{attr}(trexio_file_from)  \n\
    trexio.write_{attr}(trexio_file_to, data)    \n\
            '
            f_out.write(block)

        # Now process the sparse data
        for attr in data_sparse:
            block = f'\n\
  offset_file = 0 ; eof = False                                   \n\
  if trexio.has_{attr}(trexio_file_from):                         \n\
    while(not eof):                                               \n\
      indices, values, read_size, eof = trexio.read_{attr}(       \n\
          trexio_file_from, offset_file, buffer_size              \n\
          )                                                       \n\
      trexio.write_{attr}(                                        \n\
          trexio_file_to, offset_file, read_size, indices, values \n\
          )                                                       \n\
      offset_file += read_size                                    \n\
            '
            f_out.write(block)

        # Finally process the determinant data
        for attr in data_determinant:
            if 'determinant_num' in attr:
                continue
            block = f'\n\
  offset_file = 0 ; eof = False                                   \n\
  if trexio.has_{attr}(trexio_file_from):                         \n\
    while(not eof):                                               \n\
      data, read_size, eof = trexio.read_{attr}(                  \n\
          trexio_file_from, offset_file, buffer_size              \n\
          )                                                       \n\
      trexio.write_{attr}(                                        \n\
          trexio_file_to, offset_file, read_size, data            \n\
          )                                                       \n\
      offset_file += read_size                                    \n\
            '
            f_out.write(block)


def run_converter(filename_from, filename_to, back_end_to, back_end_from=None, overwrite=None):
    """The high-level converter function."""

    import os
    try:
        import trexio
    except ImportError as exc:
        raise ImportError("trexio Python package is not installed.") from exc

    try:
        # For proper python package
        from converters.converter_generated import data_handler
    except:
        try:
            # For argparse-based CLI
            from converter_generated import data_handler
        except:
            raise ImportError('The generated (JSON-based) data_handler.py module cannot be imported.')

    if not os.path.exists(filename_from):
        raise FileNotFoundError(f'Input file {filename_from} not found.')

    if os.path.exists(filename_to):
        if overwrite:
            if '*' in filename_to:
                raise ValueError('Are you sure?')
            else:
                os.system(f'rm -rf -- {filename_to}')
        else:
            raise ValueError(f'Output file {filename_to} already exists. Consider using the `-w` argument.')

    with trexio.File(filename_from, 'r', back_end_from) as trexio_file_from:
        with trexio.File(filename_to, 'w', back_end_to) as trexio_file_to:
            data_handler(trexio_file_from, trexio_file_to)


def run(filename_from, filename_to, back_end_to, back_end_from=None, overwrite=None, json_filename='trex.json') -> None:
    """Interface to the upstream master script."""

    generate_converter(json_filename)
    run_converter(filename_from, filename_to, back_end_to, back_end_from, overwrite)

    print(f'\n\
        Conversion from {filename_from} to {filename_to} is finished.               \n\
        Note: conversion of the CI coefficients is performed only for ground state. \n\
        ')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument(
        '--json_file',
        type=str,
        nargs=1,
        default='trex.json',
        help='TREXIO configuration file (in JSON format).'
        )
    parser.add_argument(
        '--file_from',
        type=str,
        nargs=1,
        required=True,
        help='Input TREXIO file name.'
        )
    parser.add_argument(
        '--file_to',
        type=str,
        nargs=1,
        required=True,
        help='Output TREXIO file name.'
        )
    parser.add_argument(
        '--back_end_to',
        type=int,
        nargs=1,
        required=True,
        help='Output TREXIO back end.'
        )
    parser.add_argument(
        '--back_end_from',
        type=int,
        nargs=1,
        help='Input TREXIO back end.'
        )
    parser.add_argument(
        '--overwrite',
        type=bool,
        nargs=1,
        default=True,
        help='Overwrite flag. Default: True.'
        )

    args = parser.parse_args()

    json_filename = args.json_file[0]
    filename_from = args.file_from[0]
    filename_to   = args.file_to[0]
    back_end_to   = args.back_end_to[0]
    back_end_from = args.back_end_from[0] if isinstance(args.back_end_from, list) else args.back_end_from
    overwrite     = args.overwrite[0] if isinstance(args.overwrite, list) else args.overwrite

    run(filename_from, filename_to, back_end_to, back_end_from, overwrite, json_filename)
