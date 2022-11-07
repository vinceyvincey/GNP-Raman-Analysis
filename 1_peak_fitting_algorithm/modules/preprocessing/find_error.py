"""
Find errors
"""

import numpy as np

def standard_deviation_regions(wn, y,region1=1200, region2=[1700,2250],region3=3050):
    """
    Find the standard deviation of 11 sequential data points in three different regions:
        1. x < 1200
        2. 1700 < x < 2250 
        3. x > 3050
    """

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    # REGION 1
    target_region_1 = y[(wn < region1)]
    # print(target_region_1)
    region_1_SD = []
    for group in chunker(target_region_1, 11):
        each_chunk = np.array(group)
        each_chunk_SD = np.std(each_chunk)
        region_1_SD.append(each_chunk_SD)

    # REGION 2
    target_region_2 = y[(wn > region2[0]) & (wn < region2[1])]
    region_2_SD = []
    for group in chunker(target_region_2, 11):
        each_chunk = np.array(group)
        each_chunk_SD = np.std(each_chunk)
        region_2_SD.append(each_chunk_SD)

    # REGION 3
    target_region_3 = y[(wn > region3)]
    region_3_SD = []
    for group in chunker(target_region_3, 11):
        each_chunk = np.array(group)
        each_chunk_SD = np.std(each_chunk)
        region_3_SD.append(each_chunk_SD)

    all_regions = np.concatenate((region_1_SD, region_2_SD, region_3_SD))
    return np.mean(all_regions)