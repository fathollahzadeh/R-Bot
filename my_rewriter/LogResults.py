import pandas as pd


class LogLLMResults(object):
    def __init__(self,
                 llm_model: str,
                 time_total: float,
                 all_token_count: int = 0,
                 dataset_name: str = '',
                 dbms: str = '',
                 query_id: str = '',
                 ):
        self.llm_model = llm_model
        self.time_total = time_total
        self.all_token_count = all_token_count
        self.dataset_name = dataset_name
        self.dbms = dbms
        self.query_id = query_id
        self.columns = ["dataset_name", "dbms", "llm_model","query_id", "time_total", "all_token_count"]

    def save_results(self, result_output_path: str):

        try:
            df_result = pd.read_csv(result_output_path)

        except Exception as err:
            df_result = pd.DataFrame(columns=self.columns)

        df_result.loc[len(df_result)] = [self.dataset_name, self.dbms , self.llm_model, self.query_id, self.time_total, self.all_token_count]
        df_result.to_csv(result_output_path, index=False)


def save_llm_log(llm_model,result_log_path, time_total, all_token_count, dataset_name, dbms, query_id):
    log_results = LogLLMResults(llm_model=llm_model,time_total=time_total, all_token_count=all_token_count,
                                dataset_name=dataset_name, dbms=dbms, query_id=query_id)
    log_results.save_results(result_output_path=result_log_path)

