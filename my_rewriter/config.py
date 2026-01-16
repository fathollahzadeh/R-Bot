from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
import yaml
from .FileHandler import read_text_file_line_by_line
from my_rewriter.LLM_API_Key import LLM_API_Key

CACHE_PATH = None #'/home/saeed/Downloads/RBoth/cache/'
CASE_RULES_PATH = 'stackoverflow-rewrite-rules-query-optimization.jsonl'

_dbms = None
_dataset_name = None
_workload = []
_llm_model = None
_llm_platform = None
_delay = None
_temperature = None
_top_p = None
_top_k = None
_last_API_Key = None
_LLM_API_Key = None
_system_log_file = None

_pgsql_user = None
_pgsql_password = None
_pgsql_host = None
_pgsql_port = None

_output_path = None
_result_log_path = None
_workload_output = None


def init_llms(model_type: str = '', load_model=True) -> dict[str, str]:
    # embed_dim = 1536
    if 'gemini' in model_type or "gpt-oss" in model_type:
        if load_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name='Alibaba-NLP/gte-Qwen2-1.5B-instruct',
                max_length=131072,
                device="cpu"
            )
        embed_dim = 1536
    else:
        if load_model:
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-3-small"
            )
        embed_dim = 1536

    if 'gemini' in model_type.lower():
        if load_model:
           Settings.llm = Gemini(api_key=_last_API_Key, model=model_type)
    elif "gpt-oss" in model_type.lower():
        if load_model:
            Settings.llm = Groq(api_key=_last_API_Key, model=f"openai/{model_type}")
    else:
        print(f" -- Model ({model_type}) is not support! -- ")
        raise  # Re-raises the ZeroDivisionError

    from my_rewriter.prompts import GEN_CASE_REWRITE_SYS_PROMPT, GEN_CASE_REWRITE_USER_PROMPT, SELECT_CASE_RULE_SYS_PROMPT, SELECT_CASE_RULE_USER_PROMPT, CLUSTER_REWRITE_SYS_PROMPT, CLUSTER_REWRITE_USER_PROMPT, SUMMARIZE_REWRITE_SYS_PROMPT, SUMMARIZE_REWRITE_USER_PROMPT, SELECT_RULES_SYS_PROMPT, SELECT_RULES_USER_PROMPT, ARRANGE_RULE_SETS_SYS_PROMPT, ARRANGE_RULE_SETS_USER_PROMPT, ARRANGE_RULES_SYS_PROMPT, ARRANGE_RULES_USER_PROMPT, REARRANGE_RULES_SYS_PROMPT, REARRANGE_RULES_USER_PROMPT, SELECT_ARRANGE_RULES_SYS_PROMPT, SELECT_ARRANGE_RULES_USER_PROMPT, RAG_SELECT_ARRANGE_RULES_SYS_PROMPT, RAG_SELECT_ARRANGE_RULES_USER_PROMPT
    return {
        'GEN_CASE_REWRITE_SYS_PROMPT': GEN_CASE_REWRITE_SYS_PROMPT,
        'GEN_CASE_REWRITE_USER_PROMPT': GEN_CASE_REWRITE_USER_PROMPT,
        'SELECT_CASE_RULE_SYS_PROMPT': SELECT_CASE_RULE_SYS_PROMPT,
        'SELECT_CASE_RULE_USER_PROMPT': SELECT_CASE_RULE_USER_PROMPT,
        'CLUSTER_REWRITE_SYS_PROMPT': CLUSTER_REWRITE_SYS_PROMPT,
        'CLUSTER_REWRITE_USER_PROMPT': CLUSTER_REWRITE_USER_PROMPT,
        'SUMMARIZE_REWRITE_SYS_PROMPT': SUMMARIZE_REWRITE_SYS_PROMPT,
        'SUMMARIZE_REWRITE_USER_PROMPT': SUMMARIZE_REWRITE_USER_PROMPT,
        'SELECT_RULES_SYS_PROMPT': SELECT_RULES_SYS_PROMPT,
        'SELECT_RULES_USER_PROMPT': SELECT_RULES_USER_PROMPT,
        'ARRANGE_RULE_SETS_SYS_PROMPT': ARRANGE_RULE_SETS_SYS_PROMPT,
        'ARRANGE_RULE_SETS_USER_PROMPT': ARRANGE_RULE_SETS_USER_PROMPT,
        'ARRANGE_RULES_SYS_PROMPT': ARRANGE_RULES_SYS_PROMPT,
        'ARRANGE_RULES_USER_PROMPT': ARRANGE_RULES_USER_PROMPT,
        'REARRANGE_RULES_SYS_PROMPT': REARRANGE_RULES_SYS_PROMPT,
        'REARRANGE_RULES_USER_PROMPT': REARRANGE_RULES_USER_PROMPT,
        'SELECT_ARRANGE_RULES_SYS_PROMPT': SELECT_ARRANGE_RULES_SYS_PROMPT,
        'SELECT_ARRANGE_RULES_USER_PROMPT': SELECT_ARRANGE_RULES_USER_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_SYS_PROMPT': RAG_SELECT_ARRANGE_RULES_SYS_PROMPT,
        'RAG_SELECT_ARRANGE_RULES_USER_PROMPT': RAG_SELECT_ARRANGE_RULES_USER_PROMPT,
        'EMBED_DIM': embed_dim
    }

def init_db_config(database: str) -> dict[str, str]:
    return {'host':  _pgsql_host,
            'port': _pgsql_port,
            'user': _pgsql_user,
            'password': _pgsql_password,
            'dbname': database,
            'db': 'postgresql'
    }

def load_config_system(system_log: str,
                       llm_model: str = None,
                       config_path: str = "LLMConfig.yaml",
                       api_config_path: str = "APIKeys.yaml",
                       CA_PATH: str = None,
                       workload_path: str = None,
                       db_config_path: str = "DBConfig.yaml",
                       output_path: str=None,
                       result_log_path:str = None,
                       workload_output:str = None,
                       dataset_name:str = None,
                       dbms:str = None,):
    import yaml
    global _llm_model
    global _llm_platform
    global _max_token_limit
    global _max_out_token_limit
    global _delay
    global _last_API_Key
    global _LLM_API_Key
    global _system_log_file
    global _temperature
    global _top_k
    global _top_p
    global _dataset_name
    global CACHE_PATH
    global _output_path
    global _result_log_path
    global _workload_output
    global _dbms

    _output_path = output_path
    _result_log_path = result_log_path
    _workload_output = workload_output
    _dbms = dbms
    _dataset_name = dataset_name

    _system_log_file = system_log
    CACHE_PATH = CA_PATH
    with open(config_path, "r") as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("llm_platform")
                try:
                    if conf.get(llm_model) is not None:
                        _llm_model = llm_model
                        _llm_platform = plt

                        try:
                            _max_token_limit = int(conf.get(llm_model).get('token_limit'))
                        except:
                            _max_token_limit = 4000

                        try:
                            _max_out_token_limit = int(conf.get(llm_model).get('max_output_tokens'))
                        except:
                            _max_out_token_limit = 4000

                        try:
                            _delay = int(conf.get(llm_model).get('delay'))
                        except:
                            _delay = 10

                        try:
                            _temperature = float(conf.get(llm_model).get('temperature'))
                        except:
                            _temperature = 0

                        try:
                            _top_k = int(conf.get(llm_model).get('top_k'))
                        except:
                            _top_k = 64

                        try:
                            _top_p = float(conf.get(llm_model).get('top_p'))
                        except:
                            _top_p = 0.95

                        break
                except Exception as ex:
                    pass

        except yaml.YAMLError as ex:
            raise Exception(ex)

        if _llm_model is None:
            raise Exception(f'Error: model "{llm_model}" is not in the Config.yaml list!')

        (_, _last_API_Key) = LLM_API_Key(api_config_path=api_config_path).get_API_Key()


        if workload_path is not None:
            load_workload_queries(workload_path=workload_path)

        load_db_config(db_config_path=db_config_path)



def load_db_config(db_config_path: str = "DBConfig.yaml"):
    global _pgsql_user
    global _pgsql_password
    global _pgsql_host
    global _pgsql_port

    with open(db_config_path, "r") as f:
        try:
            configs = yaml.load(f, Loader=yaml.FullLoader)
            for conf in configs:
                plt = conf.get("database")
                if plt == "Postgres":
                    try:
                        _pgsql_user = conf.get("user")
                        _pgsql_password = conf.get("password")
                        _pgsql_host = conf.get("host")
                        _pgsql_port = conf.get("port")
                    except:
                        pass

        except yaml.YAMLError as ex:
            raise Exception(ex)

def load_workload_queries(workload_path: str):
    global _workload
    for path, subdirs, files in os.walk(workload_path):
        for name in files:
            if name.endswith(".sql"):
                head, tail = os.path.split(name)
                fname = os.path.join(path, name)
                query = read_text_file_line_by_line(fname)
                if "SELECT" in query or "select" in query:
                    _workload.append((query, tail.replace(".sql", "")))
    return _workload