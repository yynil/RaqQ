import sys
#export HF_ENDPOINT=https://hf-mirror.com
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
RWKV_PATH = os.environ.get('RWKV_PATH')
sys.path.append(RWKV_PATH)
from helpers import start_proxy, ServiceWorker
from infer.encoders import BiCrossFusionEncoder
from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
import sqlitedict
import torch
from FlagEmbedding import BGEM3FlagModel,FlagReranker

class LLMService:
    def __init__(self,base_model_file,be_lora_file,ce_lora_file,chat_lora_file,chat_pissa_file,be_pissa_file,tokenizer_file,device="cuda",
                 crossencoder_lora_r=8,
                 crossencoder_lora_alpha=32,
                 biencoder_lora_r=8,
                 biencoder_lora_alpha=8,
                 chat_lora_r=8,
                 chat_lora_alpha=8,
                 crossencoder_targets=['emb','ffn.key','ffn.value','ffn.receptance'],
                 biencoder_targets=['att','ffn'],
                 chat_targets=['att','ffn'],
                 pooling_type='lasttoken'
                 ) -> None:
        self.device = device
        tokenizer = TRIE_TOKENIZER(tokenizer_file)
        self.tokenizer = tokenizer
        self.bgem3 = BGEM3FlagModel('BAAI/bge-m3',  
                       use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        self.reranker = FlagReranker('/media/yueyulin/data_4t/models/bge-reranker-v2-m3/', use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
        fused_model = BiCrossFusionEncoder(
            base_model_file,
            be_lora_file,
            ce_lora_file,
            chat_lora_file,
            tokenizer,
            crossencoder_lora_r,
            crossencoder_lora_alpha,
            biencoder_lora_r,
            biencoder_lora_alpha,
            chat_lora_r,
            chat_lora_alpha,
            crossencoder_targets,
            biencoder_targets,
            chat_targets,
            chat_pissa_path=chat_pissa_file,
            bi_pissa_path=be_pissa_file,
            biencoder_pooling_type=pooling_type,
            device=device
        )
        self.fused_model = fused_model
    def get_embeddings(self,inputs):
        if isinstance(inputs,str):
            inputs = [inputs]
        print(inputs)

        embeddings_1 = self.bgem3.encode(inputs, 
                                    batch_size=12, 
                                    max_length=512, # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
                                    )['dense_vecs'].tolist()
        # outputs = self.fused_model.encode_texts(inputs)
        print(embeddings_1)
        return embeddings_1
        
    def get_cross_scores(self,texts_0,texts_1):
        assert isinstance(texts_0,list) and isinstance(texts_1,list)
        # outputs = self.fused_model.cross_encode_texts(texts_0,texts_1)
        outputs = self.reranker.compute_score(
            [[texts_0[i],texts_1[i]] for i in range(len(texts_0))],
        )
        return outputs
        
    def generate_texts(self, instruction,input_text, token_count=100):
        results = self.fused_model.sampling_generate(
                instruction,
                input_text,
                token_count=token_count,
                top_k=0,
                top_p=0,
        )
        return results  
    
    def beam_generate_texts(self, instruction,input_text, token_count=100):
        out_str = self.fused_model.beam_generate(
                instruction,
                input_text,
                token_count=token_count
        )
        return out_str

class ServiceWorker(ServiceWorker):
    def init_with_config(self, config):
        base_model_file = config["base_model_file"]
        bi_lora_path = config["bi_lora_path"]
        cross_lora_path = config["cross_lora_path"]
        tokenizer_file = config["tokenizer_file"]
        chat_lora_path = config.get("chat_lora_path",None)
        chat_pissa_path = config.get("chat_pissa_path",None)
        be_pissa_path = config.get("be_pissa_path",None)
        chat_lora_r = config.get("chat_lora_r",64)
        chat_lora_alpha = config.get("chat_lora_alpha",64)
        crossencoder_lora_r = config.get("crossencoder_lora_r",8)
        crossencoder_lora_alpha = config.get("crossencoder_lora_alpha",32)
        biencoder_lora_r = config.get("biencoder_lora_r",8)
        biencoder_lora_alpha = config.get("biencoder_lora_alpha",8)
        chat_targets = config.get("chat_targets",['att','ffn'])
        be_targets = config.get("be_targets",['att','ffn'])
        ce_targets = config.get("ce_targets",['emb','ffn.key','ffn.value','ffn.receptance'])
        pooling_type = config.get("pooling_type",'lasttoken')
        device = config.get("device","cuda")
        self.llm_service = LLMService(
            base_model_file,
            bi_lora_path,
            cross_lora_path,
            chat_lora_path,
            chat_pissa_path,
            be_pissa_path,
            tokenizer_file,
            biencoder_lora_alpha=biencoder_lora_alpha,
            biencoder_lora_r=biencoder_lora_r,
            crossencoder_lora_alpha=crossencoder_lora_alpha,
            crossencoder_lora_r=crossencoder_lora_r,
            chat_lora_alpha=chat_lora_alpha,
            chat_lora_r=chat_lora_r,
            chat_targets=chat_targets,
            biencoder_targets=be_targets,
            crossencoder_targets=ce_targets,
            pooling_type=pooling_type,
            device=device)
    
    def process(self, cmd):
        print(f'LLM Service received command {cmd}')
        if cmd['cmd'] == 'GET_EMBEDDINGS':
            texts = cmd["texts"]
            value = self.llm_service.get_embeddings(texts)
            return value
        elif cmd['cmd'] == 'GET_CROSS_SCORES':
            texts_0 = cmd["texts_0"]
            texts_1 = cmd["texts_1"]
            value = self.llm_service.get_cross_scores(texts_0,texts_1)
            return value
        elif cmd['cmd'] == 'GENERATE_TEXTS':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            token_count = cmd.get("token_count",100)
            value = self.llm_service.generate_texts(instruction,input_text,token_count)
            return value
        elif cmd['cmd'] =='BEAM_GENERATE':
            instruction = cmd["instruction"]
            input_text = cmd["input_text"]
            token_count = cmd.get("token_count",100)
            value = self.llm_service.beam_generate_texts(instruction,input_text,token_count)
            return value

        return ServiceWorker.UNSUPPORTED_COMMAND


if __name__ == '__main__':
    base_model_file = "/media/yueyulin/KINGSTON/models/rwkv6/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth"
    bi_lora_path = '/media/yueyulin/data_4t/models/pissa_biencoder/trainable_model/epoch_0_step_160000/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    cross_lora_path = '/media/yueyulin/data_4t/models/lora/cross_encoder/epoch_0/RWKV-x060-World-1B6-v2_rwkv_lora.pth'
    chat_lora_path = '/media/yueyulin/KINGSTON/tmp/pissa_sft_drcd/20240530-112010/trainable_model/epoch_0/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth.pth'
    chat_pissa_path = '/media/yueyulin/KINGSTON/tmp/pissa_sft_drcd/20240530-112010/init_pissa.pth'
    be_pissa_path = '/media/yueyulin/data_4t/models/pissa_biencoder/init_pissa.pth'
    tokenizer_file = '/home/yueyulin/github/RWKV_LM_EXT/tokenizer/rwkv_vocab_v20230424.txt'
    llm_service = LLMService(
        base_model_file,
        bi_lora_path,
        cross_lora_path,
        chat_lora_path,
        chat_pissa_path,
        be_pissa_path,
        tokenizer_file,
        chat_lora_alpha=64,
        chat_lora_r=64)
    texts = ['我打算取消订单','我要取消订单','我要退货','我要退款']
    outputs = llm_service.get_embeddings(texts)
    print(outputs)

    from sentence_transformers.util import pairwise_cos_sim
    for qid in range(len(texts)):
        query = outputs[qid]
        for i in range(len(texts)):
            if i != qid:
                print(f'{texts[qid]} vs {texts[i]} is {pairwise_cos_sim(torch.tensor([query]),torch.tensor([outputs[i]]))}')

        print('-----------------------')


    texts_0 = ['我打算取消订单','我打算取消订单','我打算取消订单','我打算取消订单']
    texts_1 = ['我要取消订单','我要退款','我要退货','我打算取消订单']
    outputs = llm_service.get_cross_scores(texts_0,texts_1)
    print(outputs)

    instruction ='根据给定的短文，回答以下问题： 《庆余年》是谁主演的？'
    input_text = '《庆余年》是由腾讯影业、新丽电视、天津深蓝影视、上海阅文影视等出品，王倦、猫腻担任编剧，孙皓执导，张若昀、李沁领衔主演，陈道明、吴刚、李小冉、辛芷蕾、李纯、宋轶联合主演的古装剧。身世神秘的少年——范闲，自小跟随奶奶生活在海边小城澹州，随着一位老师的突然造访，他看似平静的生活开始直面重重的危机与考验。在神秘老师和一位蒙眼守护者的指点下，范闲熟识药性药理，修炼霸道真气并精进武艺，而后接连化解了诸多危局。因对身世之谜的好奇，范闲离开澹州，前赴京都。 在京都，范闲饱尝人间冷暖并坚守对正义、良善的坚持，历经家族、江湖、庙堂的种种考验与锤炼，书写了光彩的人生传奇。。该剧于2019年11月26日在腾讯视频、爱奇艺首播；该剧播出后斩获了第26届上海电视节最佳中国电视剧提名，第九届大学生电视节大学生赏析推荐电视剧以及2020年度优秀海外传播作品。'
    print(llm_service.generate_texts(instruction,input_text,token_count=100))


    print(llm_service.beam_generate_texts(instruction,input_text,token_count=100))