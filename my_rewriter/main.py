import sys
import os
from argparse import ArgumentParser

sys.path.append('..')
from my_rewriter.config import init_llms, init_db_config, load_config_system
from my_rewriter.database import DBArgs
from my_rewriter.test_utils import test
from my_rewriter.rag_retrieve import init_docstore

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--index', type=str, default='hybrid')
    parser.add_argument('--topk', type=int, default=10)

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

if __name__ == '__main__':
    args = parse_arguments()
    load_config_system(system_log=args.system_log, llm_model=args.llm_model, CA_PATH=args.cache_path,
                       workload_path=args.workload_path, result_log_path=args.result_log_path, output_path=args.output_path,
                       workload_output=args.workload_output, dbms=args.dbms, dataset_name=args.database)

    from my_rewriter.config import _workload

    model_args = init_llms(args.llm_model)
    pg_config = init_db_config(args.database)

    RETRIEVER_TOP_K = args.topk
    CASE_BATCH = 5
    RULE_BATCH = 10
    REWRITE_ROUNDS = 1

    pg_args = DBArgs(pg_config)
    schema_path = os.path.join('../schemas', f"{args.database}.sql")
    schema = open(schema_path, 'r').read()

    docstore = init_docstore()

    for (query, name) in _workload:

        if name in ["query17"]:
            test(name, query, schema, pg_args, model_args, docstore, args.log_dir, RETRIEVER_TOP_K=RETRIEVER_TOP_K,
             CASE_BATCH=CASE_BATCH, RULE_BATCH=RULE_BATCH, REWRITE_ROUNDS=REWRITE_ROUNDS, index=args.index)
