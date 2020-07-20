#!/usr/bin/env python

import numpy as np, os, sys
from scipy.io import loadmat
from run_12ECG_classifier import load_12ECG_model, run_12ECG_classifier

def load_challenge_data(filename):

    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()


    return data, header_data


def save_challenge_predictions(output_directory,filename,scores,labels,classes):

    recording = os.path.splitext(filename)[0]
    new_file = filename.replace('.mat','.csv')
    output_file = os.path.join(output_directory,new_file)

    # Include the filename as the recording number
    recording_string = '#{}'.format(recording)
    class_string = ','.join(classes)
    label_string = ','.join(str(i) for i in labels)
    score_string = ','.join(str(i) for i in scores)

    with open(output_file, 'w') as f:
        f.write(recording_string + '\n' + class_string + '\n' + label_string + '\n' + score_string + '\n')


#     return sorted(classes)
def get_classes(input_directory,files):
    classes=['270492004','164889003','164890007','426627000','713427006','713426002','445118002','39732003',
              '164909002','251146004','698252002','10370003','284470004','427172004','164947007','111975006',
                  '164917005','47665007','59118001','427393009','426177001','426783006','427084000','63593006',
                  '164934002','59931005','17338001']   
    
    return sorted(classes)
if __name__ == '__main__':
    # Parse arguments.
    if len(sys.argv) != 4:
        raise Exception('Include the input and output directories as arguments, e.g., python driver.py input output.')

    model_input = sys.argv[1]
    input_directory = sys.argv[2]
    output_directory = sys.argv[3]

    # Find files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)

    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    classes=get_classes(input_directory,input_files)

    # Load model.
    print('Loading 12ECG model...')
    model = load_12ECG_model(model_input)

    # Iterate over files.
    print('Extracting 12ECG features...')
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print('    {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory,f)
        data,header_data = load_challenge_data(tmp_input_file)
        current_label, current_score = run_12ECG_classifier(data,header_data,classes, model)
        # Save results.
        save_challenge_predictions(output_directory,f,current_score,current_label,classes)


    print('Done.')