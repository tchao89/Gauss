# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from colorama import Fore

NNI_HOME_DIR = os.path.join(os.path.expanduser('~'), 'nni-experiments')

ERROR_INFO = 'ERROR: '
NORMAL_INFO = 'INFO: '
WARNING_INFO = 'WARNING: '

DEFAULT_REST_PORT = 8080
REST_TIME_OUT = 20

EXPERIMENT_SUCCESS_INFO = Fore.GREEN + 'Successfully started experiment!\n' + Fore.RESET + \
                          '------------------------------------------------------------------------------------\n' \
                          'The experiment id is %s\n'\
                          'The Web UI urls are: %s\n' \
                          '------------------------------------------------------------------------------------\n'

LOG_HEADER = '-----------------------------------------------------------------------\n' \
             '                Experiment start time %s\n' \
             '-----------------------------------------------------------------------\n'

EXPERIMENT_START_FAILED_INFO = 'There is an experiment running in the port %d, please stop it first or set another port!\n' \
                               'You could use \'nnictl stop --port [PORT]\' command to stop an experiment!\nOr you could ' \
                               'use \'nnictl create --config [CONFIG_PATH] --port [PORT]\' to set port!\n'

EXPERIMENT_INFORMATION_FORMAT = '----------------------------------------------------------------------------------------\n' \
                     '                Experiment information\n' \
                     '%s\n' \
                     '----------------------------------------------------------------------------------------\n'

EXPERIMENT_DETAIL_FORMAT = 'Id: %s    Name: %s    Status: %s    Port: %s    Platform: %s    StartTime: %s    EndTime: %s\n'

EXPERIMENT_MONITOR_INFO = 'Id: %s    Status: %s    Port: %s    Platform: %s    \n' \
                          'StartTime: %s    Duration: %s'

TRIAL_MONITOR_HEAD = '-------------------------------------------------------------------------------------\n' + \
                    '%-15s %-25s %-25s %-15s \n' % ('trialId', 'startTime', 'endTime', 'status') + \
                     '-------------------------------------------------------------------------------------'

TRIAL_MONITOR_CONTENT = '%-15s %-25s %-25s %-15s'

TRIAL_MONITOR_TAIL = '-------------------------------------------------------------------------------------\n\n\n'

TUNERS_SUPPORTING_IMPORT_DATA = {
    'TPE',
    'Anneal',
    'GridSearch',
    'MetisTuner',
    'BOHB',
    'SMAC',
    'BatchTuner'
}

TUNERS_NO_NEED_TO_IMPORT_DATA = {
    'Random',
    'Hyperband'
}

SCHEMA_TYPE_ERROR = '%s should be %s type!'
SCHEMA_RANGE_ERROR = '%s should be in range of %s!'
SCHEMA_PATH_ERROR = '%s path not exist!'
