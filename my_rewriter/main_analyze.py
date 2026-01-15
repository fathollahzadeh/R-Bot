from argparse import ArgumentParser
import os
from tqdm import tqdm
import time
from datetime import datetime
from prettytable import PrettyTable
import logging
import json
import typing as t
import argparse
import re
from collections import defaultdict
import sys

sys.path.append('..')
from my_rewriter.config import init_llms, init_db_config, load_config_system
from my_rewriter.test_utils import test
from my_rewriter.rag_retrieve import init_docstore
from my_rewriter.database import DBArgs, Database
from my_rewriter.db_utils import compare, actual_time, actual_time_once
from my_rewriter.my_utils import MyModel

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--compute_latency', action='store_true', required=False, help='whether to compute SQL latency')
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--large', action='store_true', required=False, help='whether to execute SQL queries on large database')
    parser.add_argument('--no_reflection', action='store_true', required=False, help='whether to reflect query rewrite')

    parser.add_argument('--llm-model', type=str, default=None)
    parser.add_argument('--system-log', type=str, default="/tmp/cat-system-log.dat")
    parser.add_argument('--output-path', type=str, default="/tmp/results.csv")
    parser.add_argument('--log-dir', type=str, default="/tmp/")
    parser.add_argument('--result-log-path', type=str, default="/tmp/results-log.csv")
    parser.add_argument('--workload-path', type=str, default=None)
    parser.add_argument('--workload-output', type=str, default="/tmp/")
    parser.add_argument('--dbms', type=str, default="postgresql")
    parser.add_argument('--cache-path', type=str, default="/tmp")

    args = parser.parse_args()
    return args


def rewrite(input_cost: float, output_cost: float, used_rules: t.List[str]) -> bool:
    return output_cost != float("inf") and (input_cost >= output_cost or 'SUB_QUERY_TO_CORRELATE' in str(used_rules))


def is_improved(t: t.Dict, idx: int, compute_latency: bool, large) -> bool:
    if rewrite(t['input_cost'], t['rewrites'][idx]['output_cost'], t['rewrites'][idx]['used_rules']):
        if compute_latency:
            if large:
                return t['input_latency'] > t['rewrites'][idx]['output_latency']
            else:
                return compare(t['input_var'], t['rewrites'][idx]['output_var'])
        else:
            return t['input_cost'] > t['rewrites'][idx]['output_cost']
    return False


def _analyze(name: str, input_latency: float, log_dir:str, compute_latency, large, no_reflection) -> dict:
    log_filename = f'{log_dir}/{name}.log'

    retrieval_start = None
    retrieval_end = None
    arrange_first_end = None
    arrange_second_start = None
    arrange_end = None
    rearrange_time = None
    rewrite_res = []
    model = MyModel(model_args=model_args, query_id=name)
    with open(log_filename, 'r') as f:
        lines = list(f.readlines())
        retrieval_start = lines[0].split(',')[0]
        retrieval_start = datetime.strptime(retrieval_start, '%H:%M:%S')
        for line in lines:
            if 'root INFO Input Cost' in line:
                retrieval_start = line.split(',')[0]
                retrieval_start = datetime.strptime(retrieval_start, '%H:%M:%S')
            elif 'root INFO Retrieved Rewrite Cases' in line:
                retrieval_end = line.split(',')[0]
                retrieval_end = datetime.strptime(retrieval_end, '%H:%M:%S')
            elif 'root INFO Intermediate Results' in line:
                arrange_first_end = line.split(',')[0]
                arrange_first_end = datetime.strptime(arrange_first_end, '%H:%M:%S')
            elif 'root INFO Start recipe-based rewrite' in line:
                arrange_second_start = line.split(',')[0]
                arrange_second_start = datetime.strptime(arrange_second_start, '%H:%M:%S')
            elif 'root INFO Arranged Rule Sequence' in line or 'root INFO Rule Sequence' in line:
                arrange_end = line.split(',')[0]
                arrange_end = datetime.strptime(arrange_end, '%H:%M:%S')
            elif 'root DEBUG {\'messages\'' in line:
                obj = eval(line[line.find('{'):])
                if obj['messages'][0]['content'] == model.REARRANGE_RULES_SYS_PROMPT:
                    rearrange_time = obj['time']
            elif 'root INFO Rewrite Execution Results' in line:
                res = eval(line[line.find('{'):])
                db = Database(pg_args)
                res['output_cost'] = db.cost_estimation(res['output_sql'])
                if res['output_cost'] == -1:
                    res['output_cost'] = float("inf")
                if compute_latency:
                    if res['output_sql'] == 'None':
                        res['output_latency'] = input_latency
                        if not large:
                            res['output_var'] = [input_latency] * 5
                    else:
                        if large:
                            output_latency = actual_time_once(res['output_sql'], pg_args, 3600)
                        else:
                            output_latency, output_var = actual_time(res['output_sql'], pg_args, 300)
                        res['output_latency'] = output_latency
                        if not large:
                            res['output_var'] = output_var
                rewrite_res.append(res)
                if len(rewrite_res) == 1 and no_reflection:
                    break

    if 'no_steps' in log_dir or no_reflection:
        assert len(rewrite_res) == 1
    else:
        assert len(rewrite_res) == 2

    retrieval_time = (retrieval_end - retrieval_start).seconds * 1000
    arrange_time = (((arrange_first_end - retrieval_end).seconds if arrange_first_end is not None else 0) + (
                arrange_end - arrange_second_start).seconds + (rearrange_time if len(rewrite_res) == 2 else 0)) * 1000
    rewrite_time = sum([res['time'] for res in rewrite_res])

    rewrite_obj = {'template': name,
                   'time': {'retrieval': retrieval_time, 'arrange': arrange_time, 'rewrite': rewrite_time},
                   'rewrites': rewrite_res}
    print(rewrite_obj)
    return rewrite_obj


def analyze(query: str, name: str, log_dir: str, compute_latency, large, no_reflection) -> dict:
    db = Database(pg_args)
    input_cost = db.cost_estimation(query)
    if input_cost == -1:
        input_cost = float("inf")
    if compute_latency:
        if large:
            input_latency = actual_time_once(query, pg_args, 3600)
        else:
            input_latency, input_var = actual_time(query, pg_args, 300)
    final_res = {'template': name, 'input_sql': query, 'input_cost': input_cost,
                 'time': {'retrieval': 0, 'arrange': 0, 'rewrite': 0}, 'rewrites': []}
    if compute_latency:
        final_res['input_latency'] = input_latency
        if not large:
            final_res['input_var'] = input_var
    rewrite_obj = _analyze(name=name, input_latency=input_latency if compute_latency else None, log_dir=log_dir, no_reflection=no_reflection, compute_latency=compute_latency, large=large)
    final_res['rewrites'].extend(rewrite_obj['rewrites'])
    final_res['time']['retrieval'] += rewrite_obj['time']['retrieval']
    final_res['time']['arrange'] += rewrite_obj['time']['arrange']
    final_res['time']['rewrite'] += rewrite_obj['time']['rewrite']
    best_idx = sorted([(i, r['output_cost']) for i, r in enumerate(final_res['rewrites'])], key=lambda x: x[1])[0][0]
    final_res['best_index'] = best_idx
    return final_res

if __name__ == '__main__':
    args = parse_arguments()
    load_config_system(system_log=args.system_log, llm_model=args.llm_model, CA_PATH=args.cache_path,
                       workload_path=args.workload_path, result_log_path=args.result_log_path, output_path=args.output_path,
                       workload_output=args.workload_output, dbms=args.dbms, dataset_name=args.database)

    from my_rewriter.config import _workload

    model_args = init_llms(args.llm_model)
    pg_config = init_db_config(args.database)

    pg_args = DBArgs(pg_config)
    schema_path = os.path.join('../schemas', f"{args.database}.sql")
    schema = open(schema_path, 'r').read()

    docstore = init_docstore()

    analyze_log_filename = f'{args.log_dir}/{args.database}.log' if not args.no_reflection else f'{args.log_dir}/{args.database}_no_reflection.log'
    logging.basicConfig(filename=analyze_log_filename,
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    template_rewrites = []

    for (query, name) in _workload:
        if name in ["query17"]:
            query = re.sub(r'--.*\n', '', query)
            rewrite_obj = analyze(query, name, args.log_dir, args.compute_latency, args.large, args.no_reflection)
            template_rewrites.append(rewrite_obj)

    input_attr = 'input_latency' if args.compute_latency else 'input_cost'
    output_attr = 'output_latency' if args.compute_latency else 'output_cost'

    table = PrettyTable(['Template', 'Input', 'Output', 'Rules'])
    for t in template_rewrites:
        if is_improved(t, t['best_index'], args.compute_latency, args.large):
            table.add_row([t['template'], t[input_attr], t['rewrites'][t['best_index']][output_attr],
                           t['rewrites'][t['best_index']]['used_rules']])
    logging.info('Improvements:\n' + str(table))
    logging.info(f'Improved {table.rowcount} out of {len(template_rewrites)} queries')

    output_latencies = [t['rewrites'][t['best_index']][output_attr] if rewrite(t['input_cost'],
                                                                               t['rewrites'][t['best_index']][
                                                                                   'output_cost'],
                                                                               t['rewrites'][t['best_index']][
                                                                                   'used_rules']) else t[input_attr] for
                        t in template_rewrites]

    average_latency = sum(output_latencies) / len(output_latencies)
    logging.info(f'Average: {average_latency}')
    median_latency = sorted(output_latencies)[len(output_latencies) // 2]
    logging.info(f'Median: {median_latency}')
    latency_90th = sorted(output_latencies)[int(0.9 * len(output_latencies))]
    logging.info(f'90th Percentile: {latency_90th}')
    latency_95th = sorted(output_latencies)[int(0.95 * len(output_latencies))]
    logging.info(f'95th Percentile: {latency_95th}')

    input_latencies = [t[input_attr] for t in template_rewrites]
    average_input_latency = sum(input_latencies) / len(input_latencies)
    logging.info(f'Average Input: {average_input_latency}')
    median_input_latency = sorted(input_latencies)[len(input_latencies) // 2]
    logging.info(f'Median Input: {median_input_latency}')
    input_latency_90th = sorted(input_latencies)[int(0.9 * len(input_latencies))]
    logging.info(f'90th Percentile Input: {input_latency_90th}')
    input_latency_95th = sorted(input_latencies)[int(0.95 * len(input_latencies))]
    logging.info(f'95th Percentile Input: {input_latency_95th}')

    average_retrieval = sum([t['time']['retrieval'] for t in template_rewrites]) / len(template_rewrites)
    average_arrange = sum([t['time']['arrange'] for t in template_rewrites]) / len(template_rewrites)
    average_rewrite = sum([t['time']['rewrite'] for t in template_rewrites]) / len(template_rewrites)
    logging.info(f'Average Retrieval Time: {average_retrieval}')
    logging.info(f'Average Arrange Time: {average_arrange}')
    logging.info(f'Average Rewrite Time: {average_rewrite}')
    logging.info(f'Average Total Time: {average_retrieval + average_arrange + average_rewrite}')

    if args.large:
        overall_latencies = [t['time']['retrieval'] + t['time']['arrange'] + t['time']['rewrite'] + o for t, o in
                             zip(template_rewrites, output_latencies)]

        average_overall_latency = sum(overall_latencies) / len(overall_latencies)
        logging.info(f'Average Overall: {average_overall_latency}')
        median_overall_latency = sorted(overall_latencies)[len(overall_latencies) // 2]
        logging.info(f'Median Overall: {median_overall_latency}')
        overall_latency_90th = sorted(overall_latencies)[int(0.9 * len(overall_latencies))]
        logging.info(f'90th Percentile Overall: {overall_latency_90th}')
        overall_latency_95th = sorted(overall_latencies)[int(0.95 * len(overall_latencies))]
        logging.info(f'95th Percentile Overall: {overall_latency_95th}')