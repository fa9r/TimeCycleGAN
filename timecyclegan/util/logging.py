import sys
import os
import csv
import datetime

import git


def log_command(log_file="command_log.txt"):
    logfile = open(log_file, "a")
    logfile.write(" ".join(sys.argv) + "\n")
    logfile.close()


def log_metrics(model_name, fid, lpips, tlp, tof, log_file="metrics.csv"):
    """log to metrics.csv with model name"""
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as file:
        fieldnames = ['name', 'fid', 'lpips', 'tlp', 'tof']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'name': model_name,
            'fid': fid,
            'lpips': lpips,
            'tlp': tlp,
            'tof': tof,
        })


def log_traininig_to_csv(hparams, log_file='train_log.csv'):
    model_name = hparams.pop('Name')
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    fieldnames = ['datetime', 'name', 'commit', 'hparams']
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'datetime': datetime.datetime.now(),
            'name': model_name,
            'commit': sha,
            'hparams': ";".join([str(key) + ":" + str(value) for key, value in hparams.items()])
        })
