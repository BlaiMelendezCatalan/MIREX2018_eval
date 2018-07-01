import os
import numpy as np
from dcase_util.containers import MetaDataContainer
from sed_eval.sound_event import SegmentBasedMetrics, EventBasedMetrics
from sed_eval.io import load_event_list
from sed_eval.util.event_list import unique_event_labels
from sed_eval import metric
from intervaltree import IntervalTree
from mp_utils import run_mp


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
        ref_file_name (str): name of the reference file. It shares the name
                             with its corresponding wav file.
        metrics (dict): contains the SegmentBasedMetrics and EventBasedMetrics
                        objects, which include the intermediate statistics (tp,
                        tn, fp, ...), for one file.
        results (dict): contains the statistics for the segment-based and the
                        event-based evaluation as well as the confusion matrix
                        for one file.
    """

    (ref_labels,
     est_labels,
     ref_event_list,
     est_event_list,
     file_name,
     time_resolution,
     t_collar,
     percentage_of_length,
     eval_onset,
     eval_offset) = args

    seg_met = SegmentBasedMetrics(est_labels, time_resolution=time_resolution)
    ev_met = EventBasedMetrics(est_labels,
                               t_collar=t_collar,
                               percentage_of_length=percentage_of_length,
                               evaluate_onset=eval_onset,
                               evaluate_offset=eval_offset)

    seg_met.evaluate(ref_event_list, est_event_list)
    ev_met.evaluate(ref_event_list, est_event_list)
    metrics = {'event_based': ev_met, 'segment_based': seg_met}

    results = {}
    results['segment_based'] = seg_met.results()
    results['event_based'] = ev_met.results()

    return (file_name, metrics, results)


def get_stats_by_file(mp_results):
    stats_by_file = {}
    for (file_name, metrics, file_res) in mp_results:
        stats_by_file[file_name] = file_res

    return stats_by_file


def get_overall_intermediate_stats(mp_results, labels):
    """
    Saves the results of each file in a stats_by_file and aggregates the
    intermediate statistics (tp, fp, fn, ...) for the whole dataset.

    Args:
        mp_results (list): contains the results of the function
                           compute_file_statistics for all files in the
                           dataset.
        labels (list): unique labels used in the estimation.

    Returns:
        stats_by_file (dict): contains the results for each file in the
                                dataset.
        overall_int_stats (dict): contains the aggregated intermediate statistics
                          for the whole dataset.
    """

    stats_by_file = {}
    overall_int_stats = {}
    overall_int_stats['segment_based'] = {}
    overall_int_stats['event_based'] = {}
    for (file_name, metrics, file_res) in mp_results:
        stats_by_file[file_name] = file_res
        for base in overall_int_stats.keys():
            for label in labels:
                if label not in overall_int_stats[base].keys():
                    overall_int_stats[base][label] = {}
                # Aggregate int_stats (tp, tn, fp, fn) for segment- and
                # event-based class-wise stats for the whole dataset.
                for stat in metrics[base].class_wise[label].keys():
                    if stat not in overall_int_stats[base][label].keys():
                        overall_int_stats[base][label][stat] = 0.
                    overall_int_stats[base][label][
                              stat] += metrics[base].class_wise[label][stat]
            # Aggregate int_stats (tp, tn, fp, fn) for segment- and
            # event-based overall stats for the whole dataset.
            if 'overall' not in overall_int_stats[base].keys():
                overall_int_stats[base]['overall'] = {}
            for stat in metrics[base].overall.keys():
                if stat not in overall_int_stats[base]['overall'].keys():
                    overall_int_stats[base]['overall'][stat] = 0.
                overall_int_stats[base]['overall'][
                          stat] += metrics[base].overall[stat]

    return overall_int_stats


def get_overall_stats(int_stats, base, label):
    """
    Computes the final segment-based and event-based statistics for the whole
    dataset using the intermediate statistics.

    Args:
        int_stats (dict): intermediate statistics of the whole dataset.
        base (str): the type of evaluation. Either segment_based or
                    event_based.
        label (str): one of the unique labels used in the estimation.

    Returns:
        stats (dict): final segment-based and event-based statistics for the
                      whole dataset.
    """

    stats = {}
    stats['precision'] = metric.precision(
                            Ntp=int_stats['Ntp'],
                            Nsys=int_stats['Nsys'])
    stats['recall'] = metric.recall(
                            Ntp=int_stats['Ntp'],
                            Nref=int_stats['Nref'])
    stats['f_measure'] = metric.f_measure(
                            precision=stats['precision'],
                            recall=stats['recall'])
    if base == 'segment_based':
        stats['sensitivity'] = metric.sensitivity(
                                Ntp=int_stats['Ntp'],
                                Nfn=int_stats['Nfn'])
        stats['specificity'] = metric.specificity(
                                Ntn=int_stats['Ntn'],
                                Nfp=int_stats['Nfp'])
        if label == 'overall':
            stats['accuracy'] = metric.accuracy(
                                        Ntp=int_stats['Ntp'],
                                        Ntn=int_stats['Ntn'],
                                        Nfp=int_stats['Nfp'],
                                        Nfn=int_stats['Nfn'])
    elif base == 'event_based':
        stats['deletion_rate'] = metric.deletion_rate(
                                    Nref=int_stats['Nref'],
                                    Ndeletions=int_stats['Nfn'])
        stats['insertion_rate'] = metric.insertion_rate(
                                    Nref=int_stats['Nref'],
                                    Ninsertions=int_stats['Nfp'])
        stats['error_rate'] = metric.error_rate(
                                deletion_rate_value=stats['deletion_rate'],
                                insertion_rate_value=stats['insertion_rate'])
        if label == 'overall':
            stats['substitution_rate'] = metric.substitution_rate(
                                    Nref=int_stats['Nref'],
                                    Nsubstitutions=int_stats['Nsubs'])

    return stats


def compute_statistics(ref_dir, est_dir, time_resolution=0.001, t_collar=0.2,
                       percentage_of_length=0.5, eval_onset=True,
                       eval_offset=True, ncpus=1):
    """
    Computes statistics for the whole dataset and for each file as well as the
    confusion matrix for the whole dataset.

    Args:
        ref_dir (str): directory of the references.
        est_dir (str): directory of the estimations.
        time_resolution (float): time interval used in the segment-basd
                                 evaluation. The comparison between reference
                                 and estimation is done by segments of this
                                 length.
        t_collar (float): time interval used in the event-based evaluation.
                          estimated events are correct if they fall inside
                          a range specified by t_collar from a reference event.
        percentage_of_length (float): percentage of the length within which the
                                      estimated offset has to be in order to be
                                      consider valid estimation (form sed_eval)
        eval_onset (bool): Use onsets in the event-based evaluation.
        eval_offset (bool): Use offets in the event-based evaluation.
        ncpus (int): Number of CPU to use.

    Returns:
        overall_stats (dict): statistics of the whole dataset.
        stats_by_file (dict): statistics of each file.
    """

    ref_list = sorted(os.listdir(ref_dir))
    est_list = sorted(os.listdir(est_dir))
    all_ref_events = MetaDataContainer()
    all_est_events = MetaDataContainer()
    args = []
    for ref, est in zip(ref_list, est_list):
        assert ref == est, ("File names do not coincide."
                            "ref: %s, est: %s") % (ref, est)
        ref_event_list = load_event_list(os.path.join(ref_dir, ref))
        est_event_list = load_event_list(os.path.join(est_dir, est))
        all_ref_events += ref_event_list
        all_est_events += est_event_list
        args.append([ref_event_list,
                     est_event_list,
                     ref,
                     time_resolution,
                     t_collar,
                     percentage_of_length,
                     eval_onset,
                     eval_offset])

    ref_labels = all_ref_events.unique_event_labels
    est_labels = all_est_events.unique_event_labels

    for arg in args:
        arg.insert(0, ref_labels)
        arg.insert(1, est_labels)

    mp_results = run_mp(compute_file_statistics, args, ncpus)

    stats_by_file = get_stats_by_file(mp_results)
    overall_int_stats = get_overall_intermediate_stats(mp_results, est_labels)

    overall_stats = {}
    for base in overall_int_stats.keys():
        overall_stats[base] = {}
        for label in overall_int_stats[base].keys():
            overall_stats[base][label] = get_overall_stats(
                                            overall_int_stats[base][label],
                                            base,
                                            label)

    return overall_stats, stats_by_file
