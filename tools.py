import numpy as np


def apply_gesture_model(ls_df, classifier):
    
    '''
    function:
    ---------
    apply_gesture_model() calls classifier for gesture prediction. for each possible event frame, first identify peaks, then extract band around peak for gesture classification.
    
    parameters:
    -----------
    @ls_df: list of potential events (each event is a dataframe).
    @classifier: pretrained gesture classifier. in this case, we'll be using a pretrained random forest (from sklearn.ensemble).
        
    '''
    
    for idx, df in enumerate(ls_df):
        
        # find the max magnitude in each frame
        peak = np.argmax(df['magnitude'])

        # then extract band around peak, which will be passed to classifier
        peak_features = df.ix[peak-5: peak + 5,:]

        print 'frame: ', idx, ' | prediction: ', pd.Series(classifier.predict(peak_features)).value_counts(normalize = True)
        print '-' * 100


def calculate_maxcrosscorrelation(reference_signal, unknown_signal):    
    
    '''
    function:
    ---------
    given a reference signal and an unknown signal, calculate the max cross correlation score. the higher the score,
    the more similar two signals are. 
    
    the max cross correlation score will be used to identify events.
    
    parameters:
    -----------
    @reference_signal: 150 unit numpy array, representing reference signal.
    @unknown_signal: 150 unit numpy array
    
    returns:
    --------
    @score: int between [0,1]; represents similarity between two curves. 
    '''
    
    # https://stackoverflow.com/questions/1289415/what-is-a-good-r-value-when-comparing-2-signals-using-cross-correlation
    x = max(np.correlate(reference_signal, reference_signal, 'full'))
    y = max(np.correlate(unknown_signal, unknown_signal, 'full'))
    z = max(np.correlate(reference_signal, unknown_signal, 'full'))
    score = (z ** 2) / float(x * y)
    
    return score



from scipy import signal
import numpy as np

def identify_peaks(df, wavelet_window, rolling_window):
    '''
    function:
    ---------
    given df with magnitude, find peaks through wavelets. peaks are used as landmarks. 
    
    parameters:
    -----------
    @wavelet_window: int, referring to width of window for wavelet calculation. a value of 250 seems to work well.
    @rolling_window: int, referring to width of window for rolling mean calculation. this is needed to shift peaks 
    to its corresponding position, since the row index does not reset.
    
    returns:
    --------
    @ls_peaks_shifted: list, referring to list of indices where peaks were found through wavelets.
    '''
    
    ls_peaks = signal.find_peaks_cwt(df['magnitude'], np.arange(1,wavelet_window))
    
    ls_peaks_shifted = map(lambda x: x + rolling_window, ls_peaks)
    
    return ls_peaks_shifted







import pandas as pd
import numpy as np

def extract_window(df, left_idx, window_size):
    '''
    function:
    ---------
    simulate a detected event for gesture prediction.
    
    parameters:
    -----------
    @df: df, representing processed features. 
    @left_idx: int, representing index. a window of window_size will be created to the right of the left_idx.
    @window_size: int, representing window size. 
    
    returns:
    --------
    @window: df, representing truncated dataframe.
    
    '''
    
    right_idx = left_idx + window_size -1
    
    window = df.ix[left_idx: right_idx,:].reset_index().drop('index', axis = 1)
    
    return window







def detect_events(idx, test_df, jerk_threshold, area_threshold, window_size):

    '''
    function:
    ---------

    given a frame, detect events. events are periods with high jerk (changes in acceleration) and consistent increasing acceleration.

    parameters:
    -----------
    @idx: starting index.
    @test_df: collection of processed features.
    @jerk_threshold: int. threshold for max(jerk) - min(jerk). deprecated as of 2/9/15.
    @area_threshold: deprecated as of 2/9/15.
    @window_size:
    '''

    window_size -= 1 # adjust by 1 for slicing operation
    
    # difference between frames and df is frames is magnitude only, whereas df contains all features
    ls_frames = []
    ls_df = []

    while idx < test_df.shape[0]:
        
        # grab frame for event detection
        frame = np.array(test_df.ix[(idx - window_size): idx,:]['magnitude'])

        
        # chomp signal to center peaks
        signal_chomped = frame[20: 30]
        
        if np.max(signal_chomped) > 1.5:
            print 'identified event at ', np.argmax(signal_chomped) + idx - window_size + 20
            
            # grab window for visual inspetion
            ls_frames.append(signal_chomped)
            
            # pass corresponding frame for gesture classification
            frame_features = test_df.ix[(idx - window_size): (idx),:].reset_index().drop('index', axis = 1)
            ls_df.append(frame_features)
            
            # skip so there is no double counting
            idx += 69
            
        # otherwise continue to slide window in 0.5 sec intervals
        else:
            idx += 9
            
    print 'number of events: ', len(ls_frames)
            
    return ls_frames, ls_df
