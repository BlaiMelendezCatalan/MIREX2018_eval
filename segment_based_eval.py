import os
import numpy as np
from dcase_util.containers import MetaDataContainer, MetaDataItem
from sed_eval.sound_event import SegmentBasedMetrics
from sed_eval.io import load_event_list
from sed_eval.util.event_list import unique_event_labels
from sed_eval import metric
from intervaltree import IntervalTree
from mp_utils import run_mp
from librosa.core import load
from tqdm import tqdm


EVAL_ONSET = True
EVAL_OFFSET = True
TIME_RESOLUTION = 0.01


def reducer(str1, str2):
    """
    Joins two strings in alphabetical order using '__'.

    Args:
        str1 (str): first string.
        str2 (str): second string.
    Returns:
        Joined strings.
    """

    strs = [str1, str2]
    strs = sorted(strs)
    return strs[1] + '__' + strs[0]


def compute_file_statistics(args):
    """
    Computes the statistics for one file.

    Args:
        args (list): necessary data. Supplied by run_mp function.

    Returns:
        file_name (str): name of the reference file. It shares the name
                         with its corresponding wav file.
        seg_met (dict): SegmentBasedMetrics object, which include the
                        intermediate statistics (tp, tn, fp, ...), for one
                        file.
        results (dict): contains the segment-based statistics of one file.
    """

    (ref_labels,
     est_labels,
     ref_event_list,
     est_event_list,
     file_name) = args

    seg_met = SegmentBasedMetrics(est_labels, time_resolution=TIME_RESOLUTION)
    seg_met.evaluate(ref_event_list, est_event_list)

    raw_res = seg_met.results()
    results = {}
    for l in ref_labels:
        results[l] = {}
        p = raw_res['class_wise'][l]['f_measure']['precision']
        r = raw_res['class_wise'][l]['f_measure']['recall']
        f = raw_res['class_wise'][l]['f_measure']['f_measure']
        results[l]['precision'] = p
        results[l]['recall'] = r
        results[l]['f_measure'] = f
    a = seg_met.overall['Ntp'] / seg_met.overall['Nref']
    results['overall'] = {'accuracy': a}

    return (file_name, seg_met, results)


def get_files_stats(mp_results):
    files_stats = {}
    for (file_name, _, file_res) in mp_results:
        files_stats[file_name] = file_res

    return files_stats


def get_overall_intermediate_stats(mp_results, labels):
    """
    Saves the results of each file in a files_stats and aggregates the
    intermediate statistics (tp, fp, fn, ...) for the whole dataset.

    Args:
        mp_results (list): contains the results of the function
                           compute_file_statistics for all files in the
                           dataset.
        labels (list): unique labels used in the reference.

    Returns:
        overall_interm_stats (dict): contains the aggregated intermediate statistics
                          for the whole dataset.
    """
    overall_interm_stats = {}
    for (_, metrics, _) in mp_results:
        for label in labels:
            if label not in overall_interm_stats.keys():
                overall_interm_stats[label] = {}
            # Aggregate int_stats (tp, tn, fp, fn) for segment-based
            # class-wise stats for the whole dataset.
            for stat in metrics.class_wise[label].keys():
                if stat not in overall_interm_stats[label].keys():
                    overall_interm_stats[label][stat] = 0.
                overall_interm_stats[label][stat] += metrics.class_wise[
                                                                label][stat]
        # Aggregate int_stats (tp, tn, fp, fn) for segment-based
        # overall stats for the whole dataset.
        if 'overall' not in overall_interm_stats.keys():
            overall_interm_stats['overall'] = {}
        for stat in metrics.overall.keys():
            if stat not in overall_interm_stats['overall'].keys():
                overall_interm_stats['overall'][stat] = 0.
            overall_interm_stats['overall'][
                      stat] += metrics.overall[stat]

    return overall_interm_stats


def get_dataset_stats(int_stats, label):
    """
    Computes the final segment-based statistics for the whole dataset using the
    intermediate statistics.

    Args:
        int_stats (dict): intermediate statistics of the whole dataset.
        label (str): one of the unique labels used in the reference or
                     'overall'.

    Returns:
        stats (dict): final segment-based statistics for the whole dataset.
    """

    stats = {}
    if label != 'overall':
        stats['precision'] = metric.precision(
                                 Ntp=int_stats['Ntp'],
                                 Nsys=int_stats['Nsys'])
        stats['recall'] = metric.recall(
                                Ntp=int_stats['Ntp'],
                                Nref=int_stats['Nref'])
        stats['f_measure'] = metric.f_measure(
                                precision=stats['precision'],
                                recall=stats['recall'])
    else:
        stats['accuracy'] = int_stats['Ntp'] / int_stats['Nref']

    return stats


def fill_event_gaps(events, duration):
    filled_events = MetaDataContainer()
    if len(events) == 0:
        filled_events.append(MetaDataItem({'onset': 0.0,
                                           'offset': duration,
                                           'event_label': '-'}))
        return filled_events
    sorted_events = sorted(events)
    last_offset = 0.0
    unannotated_segments = [[0.0, duration]]
    for i, e in enumerate(sorted_events):
        for i, seg in enumerate(unannotated_segments):
            if e.onset < seg[1] and e.offset > seg[0]:
                if seg[0] >= e.onset and seg[1] <= e.offset:
                    unannotated_segments.pop(i)
                elif seg[0] >= e.onset:
                    unannotated_segments[i][0] = e.offset
                elif seg[1] <= e.offset:
                    unannotated_segments[i][1] = e.onset
                else:
                    unannotated_segments.append([e.offset, seg[1]])
                    unannotated_segments[i][1] = e.onset
        filled_events.append(MetaDataItem({'onset': e.onset,
                                           'offset': e.offset,
                                           'event_label': e.event_label}))

    for seg in unannotated_segments:
        filled_events.append(MetaDataItem({'onset': seg[0],
                                           'offset': seg[1],
                                           'event_label': '-'}))

    return filled_events


def compute_statistics(ref_dir, est_dir, audio_dir, ncpus=1):
    """
    Computes statistics for the whole dataset and for each file.

    Args:
        ref_dir (str): directory of the references.
        est_dir (str): directory of the estimations.
        ncpus (int): Number of CPU to use.

    Returns:
        dataset_stats (dict): statistics of the whole dataset.
        files_stats (dict): statistics of each file.
    """

    ref_list = sorted(os.listdir(ref_dir))
    est_list = sorted(os.listdir(est_dir))
    all_ref_events = MetaDataContainer()
    all_est_events = MetaDataContainer()
    args = []
    for ref, est in tqdm(zip(ref_list, est_list)):
        assert ref == est, ("File names do not coincide."
                            "ref: %s, est: %s") % (ref, est)
        ref_event_list = load_event_list(os.path.join(ref_dir, ref))
        est_event_list = load_event_list(os.path.join(est_dir, est))
        audio_path = os.path.join(audio_dir, ref.replace('.txt', '.wav'))
        wav, sr = load(audio_path, sr=None)
        duration = len(wav) / sr
        ref_event_list = fill_event_gaps(ref_event_list, duration)
        est_event_list = fill_event_gaps(est_event_list, duration)
        all_ref_events += ref_event_list
        all_est_events += est_event_list
        args.append([ref_event_list,
                     est_event_list,
                     ref])

    ref_labels = all_ref_events.unique_event_labels
    est_labels = all_est_events.unique_event_labels
    if '-' in ref_labels and '-' not in est_labels:
        est_labels.append('-')
    if '-' in est_labels and '-' not in ref_labels:
        ref_labels.append('-')
    
    est_labels = all_est_events.unique_event_labels

    for l in est_labels:
        if l not in ref_labels:
            raise ValueError("Ref-est class mismatch")

    for arg in args:
        arg.insert(0, ref_labels)
        arg.insert(1, est_labels)

    mp_results = run_mp(compute_file_statistics, args, ncpus)

    files_stats = get_files_stats(mp_results)
    overall_interm_stats = get_overall_intermediate_stats(mp_results,
                                                          ref_labels)

    dataset_stats = {}
    for label in overall_interm_stats.keys():
        dataset_stats[label] = get_dataset_stats(overall_interm_stats[label],
                                                 label)

    return dataset_stats, files_stats
