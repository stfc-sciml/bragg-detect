import h5py
import numpy as np

from bragg_detect import detect_bragg_peaks, plot_peaks_over_data

if __name__ == '__main__':
    # load data
    with h5py.File('data.hdf5', 'r') as h5:
        dset = list(h5.keys())[0]
        data = h5[dset][:]

    # detect
    peaks = detect_bragg_peaks(data, large_peak_size=[10, 10, 20],
                               threshold=.28, workers=1, strategy_3d='bgm')

    # if using multiple workers, pass (filename, dsetname) instead of data
    # for better efficiency; for example:
    # peaks = detect_bragg_peaks(('data.hdf5', dset),
    #                            large_peak_size=[10, 10, 50],
    #                            threshold=.2, workers=4)

    # save peak locations
    np.savetxt('result/peak_locations.txt', peaks, fmt='%d')

    # plot image without peaks
    plot_peaks_over_data(data, plot_size=(2000, 1000, 3000),
                         vmax=(.5, 0.05, 0.05),
                         peak_sets=[], axis_on=True,
                         save_to_file='result/input_image.png')

    # plot image with peaks
    plot_peaks_over_data(data, plot_size=(2000, 1000, 3000),
                         vmax=(.5, 0.05, 0.05),
                         peak_sets=[(peaks, 'w', 'o', 3, 0)], axis_on=True,
                         save_to_file='result/peaks_on_image.png')
