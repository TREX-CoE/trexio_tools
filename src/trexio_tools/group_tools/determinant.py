#!/usr/bin/env python3

def to_determinant_list(orbital_list: list, int64_num: int) -> list:
    """Convert an input orbital_list into a set of bit fields."""

    det_list = []
    bitfield = 0
    shift    = 0

    # orbital list has to be sorted in increasing order for the bitfields to be set correctly
    orb_list_sorted = sorted(orbital_list)

    for orb in orb_list_sorted:

        if orb-shift > 64:
            # this removes the 0 bit from the beginning of the bitfield
            bitfield = bitfield >> 1
            # append a bitfield to the list
            det_list.append(bitfield)
            bitfield = 0

        modulo = int((orb-1)/64)
        shift  = modulo*64
        bitfield |= (1 << (orb-shift))

    # this removes the 0 bit from the beginning of the bitfield 
    bitfield = bitfield >> 1
    det_list.append(bitfield)
    #print('Popcounts: ', [bin(d).count('1') for d in det_list)
    #print('Bitfields: ', [bin(d) for d in det_list])
    
    bitfield_num = len(det_list)
    if bitfield_num > int64_num:
        raise Exception(f'Number of bitfields {bitfield_num} cannot be more than the int64_num {int64_num}.') 
    if bitfield_num < int64_num:
        for _ in range(int64_num - bitfield_num):
            print("Appending an empty bitfield.")
            det_list.append(0)

    return det_list

