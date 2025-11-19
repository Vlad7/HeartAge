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

    ###################################################################################################################
    ########################################## RHYTMOGRAMMA ###########################################################
    ###################################################################################################################

    rr_intervals = rr_time_series_dictionary['1083']
    rits.plot_RR_intervals_time_series(rr_intervals, first_time=40000, second_time=54000)