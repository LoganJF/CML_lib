def run_pipeline(inputs):
    # Relative imports from Logan toolbox
    import sys
    from ptsa.data.readers import JsonIndexReader
    from time import time
    sys.path.append('/home2/loganf/SecondYear/CML_lib/')
    from GetData import get_subs
    from SpectralAnalysis.RollingAverage import sliding_mean_fast
    from SpectralAnalysis.MentalChronometry import Subject
    from SpectralAnalysis.ZscoreByWholeSession import get_behavioral_events_zscoring
    from scipy.stats import zscore
    from copy import deepcopy
    from ptsa.data.TimeSeriesX import TimeSeriesX
    import numpy as np
    # from RetrievalCreationHelper import create_matched_events, DeliberationEventCreator
    # from SpectralAnalysis.RollingAverage import sliding_mean_fast
    subject, session = inputs
    experiment = 'FR1'
    s = time()
    # jr = JsonIndexReader('/protocols/r1.json')
    # session=list(jr.aggregate_values('sessions', subject=subject, experiment='FR1'))[0]

    subject = Subject(subject=subject, experiment=experiment, session=session,
                      eeg_start=-1.25, eeg_end=.25, eeg_buffer=1.,
                      bipolar=True, width=5, freqs=None, verbose=True, resampled_rate=500.,
                      save=True, outdir='/scratch/loganf/MentalChronometry_Logan/')

    print(subject.subject, subject.session)
    subject.set_events()

    if subject.mean_rec < 20:
        return

    subject.set_matched_retrieval_deliberation_events(rec_min_free_before=2000, rec_min_free_after=1500,
                                                      remove_before_recall=2000, remove_after_recall=2000,
                                                      match_tol=2000)

    # -------> Set eeg of z-score
    z_events = get_behavioral_events_zscoring(subject=subject.subject,
                                              experiment=subject.experiment,
                                              session=int(subject.session),
                                              desired_step_in_sec=60,
                                              jitter=10)

    subject.set_eeg(z_events)
    z_score_eeg = subject.eeg

    # -------> Get power of z-score
    subject.morlet(data=z_score_eeg, output='power')
    z_score_eeg = subject.power

    # Create z-score possibilites from getting mean, std and median
    z_score_eeg = z_score_eeg.mean('time')
    z_mean = z_score_eeg.mean('events')
    z_std = z_score_eeg.std('events')
    z_median = z_score_eeg.median('events')
    z_ts = TimeSeriesX.concat([z_mean, z_std, z_median], 'z_score')
    z_ts['z_score'] = np.array(['mean', 'std', 'median'])

    # ------> Set eeg behavioral events, get power and average over time
    subject.set_eeg(events=subject.matched_events)
    subject.morlet(data=subject.eeg)
    avg_windows = sliding_mean_fast(subject.power, window=.2, desired_step=.02)

    # -------> Z-score each wavelet X ch X epoch using scipy
    z_values = zscore(avg_windows, 2)
    z_windows_scipy = deepcopy(avg_windows)
    z_windows_scipy.data = z_values
    freq_bands = np.array(['theta' if freq else 'hfa' for freq in z_windows_scipy['frequency'] <= 8])
    z_windows_scipy['frequency'].data = freq_bands
    z_bands_scipy = z_windows_scipy.groupby('frequency').mean('frequency')

    # -------> Z-score each wavelet X ch X epoch
    mean = z_ts.sel(z_score='mean')
    std = z_ts.sel(z_score='std')
    z_windows = (avg_windows - mean.data[:, :, None, None]) / std.data[:, :, None, None]

    # ------> Average into bands
    freq_bands = np.array(['theta' if freq else 'hfa' for freq in z_windows['frequency'] <= 8])
    z_windows['frequency'].data = freq_bands
    z_bands_whole_sess = z_windows.groupby('frequency').mean('frequency')
    print('Total Time: {}'.format(time() - s))

    save_path = '/scratch/loganf/test_50/{}/{}_{}_{}_zscoredbands'
    z_bands_whole_sess.to_hdf(save_path.format('whole_session',
                                               subject.subject,
                                               subject.experiment,
                                               subject.session))
    z_bands_scipy.to_hdf(save_path.format('scipy',
                                          subject.subject,
                                          subject.experiment,
                                          subject.session))
    return