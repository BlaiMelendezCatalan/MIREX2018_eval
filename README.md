# MIREX2018_eval
Code based on SED_EVAL (http://tut-arg.github.io/sed_eval/index.html) for the evaluation of any event-based task. In MIREX 2018 we use it for the following tasks:
* Music detection
* Speech detection
* Music and Speech Detection
* Foregound music / Backgorund music / No music segmentation

The task is defined by the labels used in the reference files.

# Statistics
## Segment-based evaluation
In the segment-level evaluation, we compare the estimation (est) produced by the algorithms with the ground truth (gt) in segments of 10 ms. We first compute the intermediate statistics for each class C, which include:

    True Positives (TPc): gt segment’s class = C & est segment’s class = C
    False Positives (FPc): gt segment’s class != C & est segment’s class = C
    True Negatives (TNc): gt segment’s class != C & est segment’s class != C
    False Negatives (FNc): gt segment’s class = C & est segment’s class != C 

Then we report class-wise Precision, Recall and F-measure.

    Precision (Pc) = TPc / (TPc + FPc)
    Recall (Rc) = TPc / (TPc + FNc)
    F-measure (Fc) = 2 * Pc * Rc / (Pc + Rc) 

As well as the overall Accuracy:

    Accuracy = (TP + TN) / (TP + TN + FP + FN) 

Where:

    TP = sum(TPc), for every class c
    FP = sum(FPc), for every class c
    TN = sum(TNc), for every class c
    FN = sum(FNc), for every class c 

## Event-based evaluation
In the event-level evaluation, we compare the estimation (est) produced by the algorithms with the reference (ref) in terms of events. Each annotated segment of the ground truth is considered and event. We first compute the intermediate statistics for the onsets and offsets of each class C, which include:

    True Positives (TPc): an est event of class = C that starts and ends at the same temporal positions as a ref event of class = C, taking into account a tolerance time-window.
    False Positives (FPc): an est event of class = C that starts and ends at temporal positions where no ref event of class = C does, taking into account a tolerance time-window.
    False Negatives (FNc): a ref event of class = C that starts and ends at temporal positions where no est event of class = C does, taking into account a tolerance time-window. 

Then we report class-wise Precision, Recall, F-measure, Deletion Rate, Insertion Rate and Error Rate.

    Precision (Pc) = TPc / (TPc + FPc)
    Recall (Rc) = TPc / (TPc + FNc)
    F-measure (Fc) = 2 * Pc * Rc / (Pc + Rc)
    Deletion Rate (Dc) = FNc / Nc
    Insertion Rate (Ic) = FPc / Nc
    Error Rate (Ec) = Dc + Ic 

Where:

    Nc is the number of ref events of class = C. 

We also report the overall version of these statistics:

    Precision (P) = TP / (TP + FP)
    Recall (R) = TP / (TP + FN)
    F-measure (F) = 2 * P * R / (P + R)
    Deletion Rate (D) = FN / N
    Insertion Rate (I) = FP / N
    Error Rate (E) = D + I 

Where:

    TP = sum(TPc), for every class c
    FP = sum(FPc), for every class c
    TN = sum(TNc), for every class c
    FN = sum(FNc), for every class c
    N is the number of ref events. 

# Usage
    import segment_based_eval as S
    import event_based_eval as E
    S.compute_statistics(est_dir='examples/est_dir/', ref_dir='examples/ref_dir/', ncpus=1)
    E.compute_statistics(est_dir='examples/est_dir/', ref_dir='examples/ref_dir/', ncpus=1)

# Input
The reference and estimation files of each audio file in the dataset must have the same name and adjust to the following format: each row should include the onset (seconds), offset (seconds) and the class of an event/segment separated by a tab. Rows should be ordered by onset time.
## Example
    onset1\toffset1\tclass1
    onset2\toffset2\tclass2
    ...\t...\t...

# Output
A call to compute_statistics returns a tuple containing two dictionaries:
* A dictionary with the statistics for the whole dataset
* A dictionary with the statistics for each file in the dataset
## Example:
### Segment-based
    ({'music': {'f_measure': 0.8333333333333334,
        'precision': 0.8333333333333334,
        'recall': 0.8333333333333334},
    'no-music': {'f_measure': 0.75, 'precision': 0.75, 'recall': 0.75},
        'overall': {'accuracy': 0.8}},
    {'1.txt': {'music': {'f_measure': 0.8333333333333333,
        'precision': 0.7142857142857143,
        'recall': 1.0},
    'no-music': {'f_measure': 0.7499999999999999,
        'precision': 1.0,
        'recall': 0.6},
    'overall': {'accuracy': 0.8}},
    '2.txt': {'music': {'f_measure': 0.8333333333333333,
        'precision': 1.0,
        'recall': 0.7142857142857143},
    'no-music': {'f_measure': 0.7499999999999999,
        'precision': 0.6,
        'recall': 1.0},
    'overall': {'accuracy': 0.8}}})
### Event-based
    ({'music': {'deletion_rate': 0.3333333333333333,
        'error_rate': 0.6666666666666666,
        'f_measure': 0.6666666666666666,
        'insertion_rate': 0.3333333333333333,
        'precision': 0.6666666666666666,
        'recall': 0.6666666666666666},
    'no-music': {'deletion_rate': 1.0,
        'error_rate': 2.0,
        'f_measure': 0.0,
        'insertion_rate': 1.0,
        'precision': 0.0,
        'recall': 0.0},
    'overall': {'deletion_rate': 0.6,
        'error_rate': 1.2,
        'f_measure': 0.4000000000000001,
        'insertion_rate': 0.6,
        'precision': 0.4,
        'recall': 0.4}},
    {'1.txt': {'music': {'deletion_rate': 0.0,
        'error_rate': 1.0,
        'f_measure': 0.6666666666666666,
        'insertion_rate': 1.0,
        'precision': 0.5,
        'recall': 1.0},
    'no-music': {'deletion_rate': 0.9999999999999998,
        'error_rate': 3.999999999999999,
        'f_measure': 0.0,
        'insertion_rate': 2.9999999999999996,
        'precision': 0.0,
        'recall': 0.0},
    'overall': {'deletion_rate': 0.3333333333333333,
        'error_rate': 2.0,
        'f_measure': 0.4,
        'insertion_rate': 1.6666666666666667,
        'precision': 0.2857142857142857,
        'recall': 0.6666666666666666}},
    '2.txt': {'music': {'deletion_rate': 0.5,
        'error_rate': 0.5,
        'f_measure': 0.6666666666666666,
        'insertion_rate': 0.0,
        'precision': 1.0,
        'recall': 0.5},
    'no-music': {'deletion_rate': 1.0,
        'error_rate': 1.3333333333333333,
        'f_measure': 0.0,
        'insertion_rate': 0.3333333333333333,
        'precision': 0.0,
        'recall': 0.0},
    'overall': {'deletion_rate': 0.7142857142857143,
        'error_rate': 0.8571428571428572,
        'f_measure': 0.4,
        'insertion_rate': 0.14285714285714285,
        'precision': 0.6666666666666666,
        'recall': 0.2857142857142857}}})
