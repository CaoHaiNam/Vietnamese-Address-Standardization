import gradio as gr
import Siameser
import Utils
import Norm_Typing
import json
import logging
from datetime import datetime

# print(1)
std = Siameser.Siameser(stadard_scope='all')
count = 0

def standardize(raw_address):
    global count
    raw_address = Norm_Typing.norm_vietnamese_sentence_accent(raw_address)    
    std_add = std.standardize(raw_address)
    std_address = dict()
    # top_1 = std.standardize(raw_address)
    # detail_address = Utils.get_detail_address(raw_address, main_address)
    # std_address['detail address'] = detail_address
    # std_address['main address'] = main_address
    top_1, top_5 = std.get_top_k(raw_address, 5)
    count += 1
    # logging.info(f'Request {count}')
    if count % 10 == 9:
        print(f'Request: {count}')
    with open(f'logs/request_{count}.json', 'w', encoding='utf8') as f:
        json.dump({raw_address: top_1, 'time': str(datetime.now())}, f, ensure_ascii=False, indent=4)
    return top_1, top_5


demo = gr.Interface(
    # fn=test,
    fn=standardize,
    inputs=gr.Textbox(label='raw address', lines=1, placeholder="Nhập địa chỉ thô"),
    outputs=[gr.JSON(label='stadard address'), gr.JSON(label='top 5 standard addresses')],
    allow_flagging='auto', 
    title='Chuẩn hóa địa chỉ tiếng Việtttttt',
    description='Công cụ sử dụng để chuẩn hóa địa chỉ tiếng Việt. <br> \
        Nhập vào 1 câu địa chỉ thô (ví dụ ở dưới), mô hình sẽ chuẩn hóa thành địa chỉ chuẩn, dưới dạng json, gồm 2 phần: <br> \
        * Địa chỉ chi tiết (detail address): thông tin về số nhà, ngõ ngách, hẻm,... được cung cấp trong địa chỉ thô. <br> \
        * Địa chỉ chính (main address): hiển thị dưới dạng dict, gồm tối đa 3 trên 4 trường thông tin: đường/phố, phường/xã, quận/huyện, tỉnh/thành phố. <br>\
        * Trong trường hợp địa chỉ thô xuất hiện cả tên đường và phường, thì địa chỉ chính chỉ chứa tên đường mà không cần phường (vì như thế đã đủ để xác định ví trí rồi). <br>',
    examples=['1 dong khoi str., dist. 1 ,hcmc',
                '112/21 bạch đằng, p.2, tân bình, tp. hồ chí minh', 
                'văn phòng và căn hộ cao cấp licogi 13 tower , thanh xuân , hn',
                'dablend hostel, 417/2 hoà hảo, phường 5, quận 10, hồ chí minh, vietnam',
                '17-05, tower 4,the sun avenue, 28 mai chi tho, district 2, ho chi minh city' 
                ],
    # article='Contact<br>Email: chnhust1@gmail.com <br> Facebook: https://www.facebook.com/CaoHaiNamHust/'
    article='Contact<br>Email: chnhust1@gmail.com'
)

# with gr.Blocks() as demo:
#     name = gr.Textbox(label='raw address', placeholder="150 kim hoa ha noi")
#     output1 = gr.Textbox(label='standard address')
#     output2 = gr.Textbox(label='top 5 addresses')
#     greet_btn = gr.Button("standardize")
#     greet_btn.click(fn=standardize, inputs=name, outputs=[output1, output2])

demo.launch()
