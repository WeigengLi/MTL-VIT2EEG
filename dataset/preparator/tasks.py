from preparator import Preparator
from preparation_config import preparation_config as config
import numpy as np


def left_right_task_data_preparation(feature_extraction = False, verbose=False):
    # We use the antisaccade dataset to extract the data for left and right benchmark task.
    saccade = config['saccade_trigger']
    fixation = config['fixation_trigger']
    cue = config['antisaccade']['cue_trigger']

    if not feature_extraction:
        preparator = Preparator(load_directory=config['LOAD_ANTISACCADE_PATH'],
                                save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['ANTISACCADE_FILE_PATTERN'],
                                save_file_name=config['output_name'], verbose=verbose)
        preparator.extract_data_at_events(extract_pattern=[cue, saccade, fixation], name_start_time='Beginning of cue', start_time=lambda events: events['latency'],
                                                                                    name_length_time='Size blocks of 500', length_time=500,
                                                                                    start_channel=1, end_channel=129, padding=False)
    else:
        preparator = Preparator(load_directory=config['LOAD_ANTISACCADE_PATH'],
                                save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['ANTISACCADE_HILBERT_FILE_PATTERN'],
                                save_file_name=config['output_name'], verbose=verbose)
        preparator.extract_data_at_events(extract_pattern=[cue, saccade, fixation], name_start_time='At saccade on-set', start_time=lambda events: events['latency'].shift(-1),
                                                                                    name_length_time='Fixed blocks of 1', length_time=1,
                                                                                    start_channel=1, end_channel=258, padding=False)

    preparator.blocks(on_blocks=['20'], off_blocks=['30'])  # take only blocks of pro-saccade
    preparator.addFilter(name='Keep right direction', f=lambda events: (events['type'].isin(['10']) & events['type'].shift(-1).isin(saccade) & (events['end_x'].shift(-1) < 400))
                                                                       | (events['type'].isin(['11']) & events['type'].shift(-1).isin(saccade) & (events['end_x'].shift(-1) > 400)))
    preparator.addFilter(name='Keep saccade if it comes after a reasonable reaction time', f=lambda events: events['latency'].shift(-1) - events['latency'] > 50)
    preparator.addFilter(name='Keep only the ones with big enough amplitude', f=lambda events: events['amplitude'].shift(-1) > 2)
    preparator.addLabel(name='Giving label 0 for left and 1 for right', f=lambda events: events['type'].apply(lambda x: 0 if x == '10' else 1))
    preparator.run()

def direction_task_data_preparation(feature_extraction=False, verbose=False):
    # We use the 'dots' dataset for direction task
    fixation = config['fixation_trigger']
    saccade = config['saccade_trigger']
    cue = config['dots']['cue_trigger']
    end_cue = config['dots']['end_cue_trigger']

    if not feature_extraction:
        preparator = Preparator(load_directory=config['LOAD_DOTS_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['DOTS_FILE_PATTERN'], save_file_name=config['output_name'],
                                verbose=verbose)
        # no padding, but cut 500 somewhere in between
        # we are interested only on the 5-triggers (fixation 41 cue saccade fixation) and we cut 500 data points in the middle
        preparator.extract_data_at_events(extract_pattern=[fixation, end_cue, cue, saccade, fixation],
                                          name_start_time='250 timepoints before the saccade',
                                          start_time=lambda events: (events['latency'].shift(-3) - 250),
                                          name_length_time='Fixed blocks of 500', length_time=500,
                                          start_channel=1, end_channel=129, padding=False)
    else:
        preparator = Preparator(load_directory=config['LOAD_DOTS_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['DOTS_HILBERT_FILE_PATTERN'], save_file_name=config['output_name'],
                                verbose=verbose)
        # we are interested only on the 5-triggers (fixation 41 cue saccade fixation) and we cut 1 point at saccade on-set
        preparator.extract_data_at_events(extract_pattern=[fixation, end_cue, cue, saccade, fixation],
                                          name_start_time='At saccade on-set',
                                          start_time=lambda events: (events['latency'].shift(-3)),
                                          name_length_time='Fixed blocks of 1', length_time=1,
                                          start_channel=1, end_channel=258, padding=False)

    preparator.blocks(on_blocks=['55'], off_blocks=['56'])  # take only blocks 55
    preparator.ignoreEvent(name='Ignore microsaccades', f=(lambda events: events['type'].isin(saccade) & (events['amplitude'] < 2)))
    preparator.ignoreEvent(name='Ignore microfixations', f=(lambda events: events['type'].isin(fixation) & (events['duration'] < 150)))
    preparator.addFilter(name='Keep saccade if it comes after a reasonable RT', f=lambda events: events['latency'].shift(-3) - events['latency'].shift(-2) > 50)
    preparator.addLabel(name='amplitude', f=lambda events: np.sqrt((events['end_x'].shift(-3) - events['start_x'].shift(-3)) ** 2 + (events['end_y'].shift(-3) - events['start_y'].shift(-3)) ** 2))
    preparator.addLabel(name='angle', f=lambda events: np.arctan2(events['end_y'].shift(-3) - events['start_y'].shift(-3), events['end_x'].shift(-3) - events['start_x'].shift(-3)))
    preparator.run()

def direction_with_processing_speed_dataset(feature_extraction=False, verbose=False):
    # fixation, saccade, fixation
    # saccade should be bigger than 0.5 (amplitude) (Repetition)
    # We use the 'processing_speed' dataset for this direction task
    fixation = config['fixation_trigger']
    saccade = config['saccade_trigger']

    if not feature_extraction:
        preparator = Preparator(load_directory=config['LOAD_PROCESSING_SPEED_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['PROCESSING_SPEED_FILE_PATTERN'], save_file_name=config['output_name'], verbose=verbose)
        preparator.extract_data_at_events(extract_pattern=[fixation, saccade, fixation], name_start_time='at the fixation start',
                                                                                         start_time=lambda events: (events['latency']),
                                                                                         name_length_time='Fixed blocks of 500', length_time=500,
                                                                                         start_channel=1, end_channel=129, padding=True)  # we are interested only on the triplets, fixation saccade fixation
    else:
        preparator = Preparator(load_directory=config['LOAD_PROCESSING_SPEED_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['PROCESSING_SPEED_HILBERT_FILE_PATTERN'], save_file_name=config['output_name'], verbose=verbose)
        preparator.extract_data_at_events(extract_pattern=[fixation, saccade, fixation], name_start_time='at saccade onset', start_time=lambda events: (events['latency'].shift(-1)),
                                                                                         name_length_time='Size 1', length_time=1,
                                                                                         start_channel=1, end_channel=258, padding=False)  # we are interested only on saccade on-set (where the data is feature extracted by hilbert trafo)

    preparator.addFilter(name='Keep only long enough fixations1', f=lambda events:events['duration'] > 50)
    preparator.addFilter(name='Keep only big enough saccade', f=lambda events: events['amplitude'].shift(-1) > 0.5)
    preparator.addFilter(name='Keep only long enough fixations2', f=lambda events: events['duration'].shift(-2) > 50)
    preparator.addLabel(name='amplitude', f=lambda events: np.sqrt((events['end_x'].shift(-1) - events['start_x'].shift(-1)) ** 2 + (events['end_y'].shift(-1) - events['start_y'].shift(-1)) ** 2))
    preparator.addLabel(name='angle', f=lambda events: np.arctan2(events['end_y'].shift(-1) - events['start_y'].shift(-1), events['end_x'].shift(-1) - events['start_x'].shift(-1)))
    preparator.run()

def position_task_data_preparation(feature_extraction, verbose=False):
    # We use the 'dots' dataset for position task
    fixation = config['fixation_trigger']

    if not feature_extraction:
        preparator = Preparator(load_directory=config['LOAD_DOTS_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['DOTS_FILE_PATTERN'], save_file_name=config['output_name'], verbose=verbose)
         # no padding, but cut 500 somewhere in between
        preparator.extract_data_at_events(extract_pattern=[fixation], name_start_time='At fixation start', start_time=lambda events: (events['latency']),
                                                                      name_length_time='Fixed blocks of 500', length_time=500,
                                                                      start_channel=1, end_channel=129, padding=False)  # we are interested only on the fixations (at a dot)
    else:
        preparator = Preparator(load_directory=config['LOAD_DOTS_PATH'], save_directory=config['SAVE_PATH'],
                                load_file_pattern=config['DOTS_HILBERT_FILE_PATTERN'], save_file_name=config['output_name'],
                                verbose=verbose)
        preparator.extract_data_at_events(extract_pattern=[fixation], name_start_time='At fixation start',
                                          start_time=lambda events: (events['latency']),
                                          name_length_time='Fixed blocks of 1', length_time=1,
                                          start_channel=1, end_channel=258,
                                          padding=False)  # we are interested only on the fixations

    preparator.blocks(on_blocks=['55'], off_blocks=['56'])  # take only blocks 55
    preparator.addFilter(name='Keep fixation that are long enough', f=lambda events: events['duration'] >= 500)
    preparator.addLabel(name='x_position', f=lambda events: events['avgpos_x'])
    preparator.addLabel(name='y_position', f=lambda events: events['avgpos_y'])
    preparator.addLabel(name='type', f=lambda events: events['type'])
    preparator.addLabel(name='pupil_size', f=lambda events: events['avgpupilsize'])
    preparator.run()

def segmentation_task_data_preparation(feature_extraction=False, verbose=False):
    raise NotImplementedError("Segmentation task is not implemented yet.")

def main():

    if config['task'] == 'LR_task':
        if config['dataset'] == 'antisaccade':
            left_right_task_data_preparation(config['feature_extraction'])
        else:
            raise ValueError("This task cannot be prepared (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Direction_task':
        if config['dataset'] == 'dots':
            direction_task_data_preparation(config['feature_extraction'])
        elif config['dataset'] == 'processing_speed':
            direction_with_processing_speed_dataset()
        else:
            raise ValueError("This task cannot be prepared with the given dataset.")

    elif config['task'] == 'Position_task':
        if config['dataset'] == 'dots':
            position_task_data_preparation(config['feature_extraction'])
        else:
            raise ValueError("This task cannot be prepared (is not implemented yet) with the given dataset.")

    elif config['task'] == 'Segmentation_task':
        segmentation_task_data_preparation(config['feature_extraction'])

    else:
        raise NotImplementedError("Task " + config['task'] + " is not implemented yet.")


main()
