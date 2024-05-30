import PySimpleGUIQt as sg
from retreiver import retrieve_from_baike
from clients.llm_client import LLMClient
import time
llm_client = LLMClient("tcp://localhost:7781")
gecko_path = '/home/yueyulin/firefox/geckodriver'
firefox_path = '/home/yueyulin/firefox/firefox'

# All the stuff inside your window.
layout = [ [sg.Text('请输入问题：'), sg.InputText(key='输入问题')],
           [sg.Button('提交'), sg.Button('清空内容')] ,
           [sg.Multiline(key='-TABLE-',size=(320,20),visible=False,enable_events=False,enter_submits=False)],
           [sg.Multiline(key='-SELECT_TABLE-',size=(320,20),visible=False,enable_events=False,enter_submits=False)],
           [sg.Multiline(key='-ANSWER_TABLE-',size=(320,20),visible=False,enable_events=False,enter_submits=False)],
           [sg.Button('选择文本'),sg.Button('回答问题'),sg.Button('清空选择')]
        ]

# Create the Window
window = sg.Window('AI提问助手', layout,size=(1024, 760))
# Finalize the window and get the screen size
window.finalize()
screen_width, screen_height = window.get_screen_dimensions()

# Calculate the position to center the window
x = (screen_width - window.size[0]) // 2
y = (screen_height - window.size[1]) // 2

# Move the window to the center of the screen
window.move(x, y)
# Event Loop to process "events" and get the "values" of the inputs
def group_texts(texts,max_length=500):
    chunks = []
    current_chunk = ''
    for text in texts:
        text = text.strip()
        if len(current_chunk) + len(text) > max_length:
            chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk += '\n' + text
    chunks.append(current_chunk)
    chunks = [chunk.replace('\n','').replace('\t','') for chunk in chunks]
    return chunks
query = ''
candidates = []
best_candidate = ''
while True:
    event, values = window.read()

    # if user closes window or clicks cancel
    if event == sg.WIN_CLOSED or event == 'Cancel':
        break

    if event == '清空内容':
        window['输入问题'].update('')
        window['-TABLE-'].update('',visible=False)
        window['-ANSWER_TABLE-'].update('',visible=False)
        window['-SELECT_TABLE-'].update('',visible=False)
        query = ''
        candidates = []
        best_candidate = ''
    if event == '提交':
        query = values['输入问题']
        print('You entered ', query)
        candidates = retrieve_from_baike(query,gecko_path,firefox_path)
        candidates = [result for result in group_texts(candidates)]
        print(len(candidates))
        str_candidates = '\n'.join([f'{index}:{candidate}' for index,candidate in enumerate(candidates)])
        str_candidates = f'搜索结果\n{str_candidates}'
        window['-TABLE-'].update(str_candidates,text_color_for_value='blue',visible=True,append=False)
        window['-SELECT_TABLE-'].update('',visible=False)
        window['-ANSWER_TABLE-'].update('',visible=False)
    if event == '选择文本':
        query = values['输入问题']
        if len(candidates) > 0 and len(query) > 0:
            print('calculate embeddings')

            embeddings = llm_client.encode([query]+candidates)['value']
            from sentence_transformers.util import pairwise_cos_sim
            query_embedding = embeddings[0]
            candidate_embeddings = embeddings[1:]
            scores = pairwise_cos_sim([query_embedding]*len(candidate_embeddings),candidate_embeddings)
            sorted_pairs = sorted(zip(candidates,scores),key=lambda x:x[1],reverse=True)
            str_sorted_results = '\n'.join(
             [
                    f'{i+1}:{sorted_pairs[i][0]} 打分:{sorted_pairs[i][1]}' for i in range(len(sorted_pairs))
            ])
            str_sorted_results = f'\nBiEncoder搜索结果\n{str_sorted_results}'
            for i in range(len(sorted_pairs)):
                window['-SELECT_TABLE-'].update(f'{i+1}:{sorted_pairs[i][0]} 相似度:{sorted_pairs[i][1]}\n',text_color_for_value='red',append=True,visible=True)


            # cross_encoder_scores = llm_client.cross_encode([query]*len(candidates),candidates)['value']
            # sorted_pairs = sorted(zip(candidates,cross_encoder_scores),key=lambda x:x[1],reverse=True)
            
            # str_sorted_results = '\n'.join(
            #     [
            #         f'{i+1}:{sorted_pairs[i][0]} 打分:{sorted_pairs[i][1]}' for i in range(len(sorted_pairs))
            #     ])
            # str_sorted_results = f'\nCrossEncoder搜索结果\n{str_sorted_results}'
            # window['-SELECT_TABLE-'].update(str_sorted_results,append=True,text_color_for_value='red',visible=True)


            best_candidate = '\n'.join([sorted_pair[0] for sorted_pair in sorted_pairs[:1]])
            print(f'选择文本:{best_candidate}')
    if event == '回答问题':
        query = values['输入问题']
        if len(best_candidate) > 0:
            print(f'回答问题:{query} 选择文本:{best_candidate}')
            instruction = f'根据给定的短文，回答以下问题：{query}'
            input_text = best_candidate
            token_count = 100
            start = time.time()
            outputs = llm_client.generate(instruction,input_text,token_count)['value']
            end = time.time()
            str_outputs = f'回答问题:{query}\n选择文本:{best_candidate}\n回答:{outputs}\n耗时:{end-start}秒\n'
            print(str_outputs)
            start = time.time()
            beam_results = llm_client.generate_beam(instruction,input_text,token_count)['value']
            end = time.time()
            beam_results_str = '\n'.join([f'{str_result}, score:{score}, beam_idx:{beam_idx}' for str_result,score,beam_idx in beam_results])
            beam_results_str = f'Beam搜索结果\n{beam_results_str}\n耗时:{end-start}秒\n'
            window['-ANSWER_TABLE-'].update(f'{str_outputs}\n{beam_results_str}',append=True,text_color_for_value='green',visible=True)
window.close()
