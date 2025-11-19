import os
import sys

#parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(parent_dir)

import rr_intervals_time_series as rits
import paths as pt


if __name__ == '__main__':

    # Get list of files with rr_intervals time series
    rr_intervals_filenames = rits.get_list_of_files_with_rr_intervals(pt.rr_intervals_folder)

    # Extract RR intervals time series from files to dictionary with id as key and RR intervals time series as value
    rr_time_series_dictionary = rits.extract_from_files_rr_intervals_time_series(pt.rr_intervals_folder, rr_intervals_filenames)

    # Find minimum count of rr_intervals in time series of dictionary
    id, count = rits.find_minimum_rr_count(rr_time_series_dictionary)
    print("Minimum count of RR-intervals in time series: id {0}, count {1}".format(id, count))

    minimum_time_miliseconds = 300000
    minimum_time_flag = rits.check_for_minimum_time_rr_time_series(rr_time_series_dictionary, minimum_time_miliseconds)
    print("All RR-intervals time series more than {0} duration: {1}".format(minimum_time_miliseconds/1000, minimum_time_flag))
    ## 440 min count, all > 5 min