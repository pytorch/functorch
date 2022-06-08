#!/usr/bin/env python
import argparse
import json
import os
import logging


gpu_pids = []

def is_gpu_compute_event(event):
    global gpu_pids
    return "pid" in event and event["pid"] in gpu_pids and "ph" in event and event["ph"] == "X"

def get_events(filename):
    f = open(filename)
    data = json.load(f)
    events = data["traceEvents"]
    return events

def get_sorted_gpu_events(events):
    sorted_gpu_events = []
    for event in events:
        if(not is_gpu_compute_event(event)):
            continue
        sorted_gpu_events.append(event)
    return sorted(sorted_gpu_events, key=lambda x: x["ts"])

def get_sorted_gpu_mm_conv_events(events):
    def is_mm_conv_event(event):
        return "name" in event and ("gemm" in event["name"] or "conv" in event["name"])
    gpu_events = get_sorted_gpu_events(events)
    sorted_events = []
    for event in gpu_events:
        if(not is_mm_conv_event(event)):
            continue
        sorted_events.append(event)
    return sorted_events


"""
return the duration of time that the events run
overlapping times of events are not double counted
"""
def get_duration(sorted_gpu_events):
    event = sorted_gpu_events[0]
    current_end_time = event["ts"] + event["dur"]
    total_duration = event["dur"]
    for event in sorted_gpu_events[1:]:
        start_time = max(event["ts"], current_end_time)
        end_time = event["ts"] + event["dur"]
        total_duration = total_duration + max(end_time - start_time, 0)
        current_end_time = max(current_end_time, end_time)
    return total_duration


"""
get the latest endtime - earliest starttime among events
Only consider events with "cat" key and event["cat"] != "Trace"
these events are considered valid CPU or GPU events
"""
def get_total_length(events):
    valid_events = []
    for event in events:
        if "cat" in event and event["cat"] != "Trace":
            valid_events.append(event)
    valid_events = sorted(valid_events, key=lambda x: x["ts"])
    total_start = valid_events[0]["ts"]
    total_end = 0
    for event in valid_events:
        if("dur" in event):
            end_time = event["ts"] + event["dur"]
        else:
            end_time = event["ts"]
        total_end = max(total_end, end_time)
    return total_end - total_start

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--filename", "-f", action="append", help="a filename of the json file to process"
    )
    group.add_argument(
        "--folder", "-fd", action="append", help="a folder of the json files to process"
    )
    args = parser.parse_args()

    if args.filename:
        filenames = args.filename
    elif args.folder:
        filenames = []
        directory = args.folder[0]
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f) and f.endswith(".json"):
                filenames.append(f)
    else:
        print("Please provide a filename or a folder name")

    print(f"filename, GPU Utilization, MM and Conv time")

    for filename in filenames:
        try:
            events = get_events(filename)
            global gpu_pids
            for event in events:
                if "name" not in event:
                    continue
                if event["name"] == 'process_labels' and "GPU" in event["args"]["labels"]:
                    gpu_pids.append(event["pid"])
            
            total_length = get_total_length(events)
            sorted_gpu_events = get_sorted_gpu_events(events)
            utilization = get_duration(sorted_gpu_events) / total_length
            
            sorted_gpu_mm_conv_events = get_sorted_gpu_mm_conv_events(events)
            mm_conv_utilization = get_duration(sorted_gpu_mm_conv_events) / total_length

            head, tail = os.path.split(filename)
            print(f"{tail}, {utilization}, {mm_conv_utilization}")
        except:
            logging.exception(f"{filename}, ERROR")
            print(f"{filename}, ERROR")


if __name__ == "__main__":
    main()