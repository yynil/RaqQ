import sys
import os
RWKV_PATH = os.environ.get('RWKV_PATH')
sys.path.append(RWKV_PATH)
from helpers import start_proxy
from src.model_run import RWKV,create_empty_args,load_embedding_ckpt_and_parse_args,BiCrossFusionEncoder,generate,enable_lora
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
import sqlitedict
import torch


class LLMService:
    def __init__(self,base_model_file,be_lora_file,ce_lora_file,tokenizer_file,device="cuda:0") -> None:
        args = create_empty_args()
        w = load_embedding_ckpt_and_parse_args(base_model_file, args)
        model = RWKV(args)
        info = model.load_state_dict(w)
        print(f'load info {info}')
        self.device = device
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
        self.tokenizer = tokenizer
        dtype = torch.bfloat16
        self.dtype = dtype
        self.model = model.to(device=device,dtype=dtype)

        self.fusedEncoder = BiCrossFusionEncoder(model,be_lora_file,ce_lora_file,tokenizer,device=device,dtype=dtype,lora_type='lora',lora_r=8,lora_alpha=32,add_mlp=True,mlp_dim=1024,target_modules=['emb','ffn.key','ffn.value','ffn.receptance'],cross_adapter_name='cross_encoder_lora',original_cross_adapter_name='embedding_lora',bi_adapter_name='bi_embedding_lora',original_bi_adapter_name='embedding_lora',sep_token_id = 2)
        from rwkv.utils import PIPELINE_ARGS
        self.gen_args = PIPELINE_ARGS(temperature = 1.0, top_p = 0.8, top_k = 100, # top_k = 0 then ignore
                        alpha_frequency = 0.25,
                        alpha_presence = 0.25,
                        alpha_decay = 0.996, # gradually decay the penalty
                        token_ban = [], # ban the generation of some tokens
                        token_stop = [0,2], # stop generation whenever you see any token here
                        chunk_len = 256) # split input into chunks to save VRAM (shorter -> slower)
    def get_embeddings(self,inputs):
        with torch.no_grad():
            if isinstance(inputs,str):
                inputs = [inputs]
            outputs = [self.fusedEncoder.encode_texts(input).tolist() for input in inputs]
            return outputs
        
    def get_cross_scores(self,texts_0,texts_1):
        with torch.no_grad():
            assert isinstance(texts_0,list) and isinstance(texts_1,list)
            outputs = [self.fusedEncoder.cross_encode_texts(text_0,text_1).item() for text_0,text_1 in zip(texts_0,texts_1)]
            return outputs
        
    def generate_texts(self, ctx, token_count=100):
        enable_lora(self.model,enable=False)
        with torch.no_grad():
            with torch.autocast(enabled=True,device_type='cuda',dtype=self.dtype):
                out_str = generate(self.model, ctx,self.tokenizer,token_count=token_count,args=self.gen_args,device=self.device)
        enable_lora(self.model,enable=True)
        return out_str  
    

def llm_worker(base_model_file,bi_lora_path,cross_lora_path,tokenizer_file,device,backend_url,frontend_url):
    import msgpack
    llm_service = LLMService(base_model_file,bi_lora_path,cross_lora_path,tokenizer_file,device)
    import zmq
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.connect(backend_url)
    print(f"\033[91mLLM worker connected to {backend_url}\033[0m")
    while True:
        message = socket.recv()
        cmd = msgpack.unpackb(message, raw=False)
        if cmd["cmd"] == "GET_EMBEDDINGS":
            texts = cmd["texts"]
            value = llm_service.get_embeddings(texts)
            value = {"value": value,"code": 200}
            socket.send(msgpack.packb(value, use_bin_type=True))
        elif cmd["cmd"] == "GET_CROSS_SCORES":
            texts_0 = cmd["texts_0"]
            texts_1 = cmd["texts_1"]
            value = llm_service.get_cross_scores(texts_0,texts_1)
            value = {"value": value,"code": 200}
            socket.send(msgpack.packb(value, use_bin_type=True))
        elif cmd["cmd"] == "GENERATE_TEXTS":
            ctx = cmd["ctx"]
            token_count = cmd.get("token_count",100)
            value = llm_service.generate_texts(ctx,token_count)
            value = {"value": value,"code": 200}
            socket.send(msgpack.packb(value, use_bin_type=True))
        else:
            socket.send(msgpack.packb({"code": 400}, use_bin_type=True))
def start_llm_service(config):
    base_model_file = config["base_model_file"]
    bi_lora_path = config["bi_lora_path"]
    cross_lora_path = config["cross_lora_path"]
    tokenizer_file = config["tokenizer_file"]
    num_workers = config.get("num_workers",1)
    device = config.get("device","cuda:0")
    backend_url = config["back_end"]["protocol"] + "://" + config["back_end"]["host"] + ":" + str(config["back_end"]["port"])
    frontend_url = config["front_end"]["protocol"] + "://" + config["front_end"]["host"] + ":" + str(config["front_end"]["port"])    
    print(f"Starting llm service with base model file {base_model_file} and bi lora path {bi_lora_path} and cross lora path {cross_lora_path} and tokenizer file {tokenizer_file} and device {device}")
    print(f"Starting llm service with backend url {backend_url} and frontend url {frontend_url}")
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    print(f'Starting {num_workers} workers')
    for i in range(num_workers):
        process = multiprocessing.Process(target=llm_worker, args=(base_model_file,bi_lora_path,cross_lora_path,tokenizer_file,device,backend_url,frontend_url))
        process.start()
    

    import multiprocessing
    process = multiprocessing.Process(target=start_proxy, args=(frontend_url,backend_url))
    process.start()

if __name__ == '__main__':
    base_model_file = "/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    bi_lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/bi-encoder/add_mlp_in_batch_neg/epoch_0_step_200000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    cross_lora_path = '/media/yueyulin/KINGSTON/models/rwkv6/lora/cross-encoder/epoch_0_step_500000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    llm_service = LLMService(base_model_file,bi_lora_path,cross_lora_path,tokenizer_file)
    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = llm_service.get_embeddings(texts)
    print(outputs)

    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(query.unsqueeze(0),outputs[i].unsqueeze(0))}')

        print('-----------------------')


    texts_0 = ['我打算取消订单','我打算取消订单','我打算取消订单','我打算取消订单']
    texts_1 = ['我要取消订单','我要退款','我要退货','我打算取消订单']
    outputs = llm_service.get_cross_scores(texts_0,texts_1)
    print(outputs)

    ctx = 'User: 我要取消订单\nBot:'
    out_str = llm_service.generate_texts(ctx)
    print(out_str)