import os
import numpy as np
from dcase_util.containers import MetaDataContainer
from sed_eval.sound_event import EventBasedMetrics
from sed_eval.io import load_event_list
from sed_eval.util.event_list import unique_event_labels
from sed_eval import metric
from intervaltree import IntervalTree
from mp_utils import run_mp


EVAL_ONSET = True
EVAL_OFFSET = True
PERCENTAGE_OF_LENGTH = 0.


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
        ev_met (dict): contains the EventBasedMetrics object, which include the
                       intermediate statistics (tp, tn, fp, ...), for one file.
        results (dict): contains the event-based statistics of one file.
    """

    (ref_labels,
     est_labels,
     ref_event_list,
     est_event_list,
     file_name,
     t_collar) = args

    ev_met = EventBasedMetrics(est_labels,
                               t_collar=t_collar,
                               percentage_of_length=PERCENTAGE_OF_LENGTH,
                               evaluate_onset=EVAL_ONSET,
                               evaluate_offset=EVAL_OFFSET)
    ev_met.evaluate(ref_event_list, est_event_list) 

    raw_res = ev_met.results()
    results = {}
    for l in ref_labels:
        results[l] = {}
        p = raw_res['class_wise'][l]['f_measure']['precision']
        r = raw_res['class_wise'][l]['f_measure']['recall']
        f = raw_res['class_wise'][l]['f_measure']['f_measure']
        d = raw_res['class_wise'][l]['error_rate']['deletion_rate']
        i = raw_res['class_wise'][l]['error_rate']['insertion_rate']
        e = raw_res['class_wise'][l]['error_rate']['error_rate']
        results[l]['precision'] = p
        results[l]['recall'] = r
        results[l]['f_measure'] = f
        results[l]['deletion_rate'] = d
        results[l]['insertion_rate'] = i
        results[l]['error_rate'] = e
    p = raw_res['overall']['f_measure']['precision']
    r = raw_res['overall']['f_measure']['recall']
    f = raw_res['overall']['f_measure']['f_measure']
    d = raw_res['overall']['error_rate']['deletion_rate']
    i = raw_res['overall']['error_rate']['insertion_rate']
    e = raw_res['overall']['error_rate']['error_rate']
    results['overall'] = {}
    results['overall']['precision'] = p
    results['overall']['recall'] = r
    results['overall']['f_measure'] = f
    results['overall']['deletion_rate'] = d
    results['overall']['insertion_rate'] = i
    results['overall']['error_rate'] = e

    return (file_name, ev_met, results)


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
        labels (list): unique labels used in the ground truth.

    Returns:
        files_stats (dict): contains the results for each file in the
                                dataset.
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
    Computes the final event-based statistics for the whole dataset using the
    intermediate statistics.

    Args:
        int_stats (dict): intermediate statistics of the whole dataset.
        label (str): one of the unique labels used in the reference of
                     'overall'.

    Returns:
        stats (dict): final event-based statistics for the whole dataset.
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
    stats['deletion_rate'] = metric.deletion_rate(
                            Nref=int_stats['Nref'],
                            Ndeletions=int_stats['Nfn'])
    stats['insertion_rate'] = metric.insertion_rate(
                            Nref=int_stats['Nref'],
                            Ninsertions=int_stats['Nfp'])
    stats['error_rate'] = metric.error_rate(
                            deletion_rate_value=stats['deletion_rate'],
                            insertion_rate_value=stats['insertion_rate'])

    return stats


def compute_statistics(ref_dir, est_dir, t_collar=0.5, ncpus=1):
    """
    Computes statistics for the whole dataset and for each file.

    Args:
        ref_dir (str): directory of the references.
        est_dir (str): directory of the estimations.
        t_collar (float): time interval used in the event-based evaluation.
                          estimated events are correct if they fall inside
                          a range specified by t_collar from a reference event.
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
                     t_collar])

    ref_labels = all_ref_events.unique_event_labels
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
