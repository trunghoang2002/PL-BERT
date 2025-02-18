import glob
import torch
import os
from collections import OrderedDict
import unicodedata
import pyopenjtalk
# import jaconv

def scan_checkpoint(cp_dir):
    pattern = os.path.join(cp_dir)
    cp_list = glob.glob(pattern)
    print(cp_list)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

kana_mapper = OrderedDict([
    ("ゔぁ","bˈa"),
    ("ゔぃ","bˈi"),
    ("ゔぇ","bˈe"),
    ("ゔぉ","bˈo"),
    ("ゔゃ","bˈʲa"),
    ("ゔゅ","bˈʲɯ"),
    ("ゔゃ","bˈʲa"),
    ("ゔょ","bˈʲo"),

    ("ゔ","bˈɯ"),

    ("あぁ","aː"),
    ("いぃ","iː"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","ɯː"),
    ("えぇ","eː"),
    ("おぉ","oː"),
    ("かぁ","kˈaː"),
    ("きぃ","kˈiː"),
    ("くぅ","kˈɯː"),
    ("くゃ","kˈa"),
    ("くゅ","kˈʲɯ"),
    ("くょ","kˈʲo"),
    ("けぇ","kˈeː"),
    ("こぉ","kˈoː"),
    ("がぁ","gˈaː"),
    ("ぎぃ","gˈiː"),
    ("ぐぅ","gˈɯː"),
    ("ぐゃ","gˈʲa"),
    ("ぐゅ","gˈʲɯ"),
    ("ぐょ","gˈʲo"),
    ("げぇ","gˈeː"),
    ("ごぉ","gˈoː"),
    ("さぁ","sˈaː"),
    ("しぃ","ɕˈiː"),
    ("すぅ","sˈɯː"),
    ("すゃ","sˈʲa"),
    ("すゅ","sˈʲɯ"),
    ("すょ","sˈʲo"),
    ("せぇ","sˈeː"),
    ("そぉ","sˈoː"),
    ("ざぁ","zˈaː"),
    ("じぃ","dʑˈiː"),
    ("ずぅ","zˈɯː"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ぜぇ","zˈeː"),
    ("ぞぉ","zˈeː"),
    ("たぁ","tˈaː"),
    ("ちぃ","tɕˈiː"),
    ("つぁ","tsˈa"),
    ("つぃ","tsˈi"),
    ("つぅ","tsˈɯː"),
    ("つゃ","tɕˈa"),
    ("つゅ","tɕˈɯ"),
    ("つょ","tɕˈo"),
    ("つぇ","tsˈe"),
    ("つぉ","tsˈo"),
    ("てぇ","tˈeː"),
    ("とぉ","tˈoː"),
    ("だぁ","dˈaː"),
    ("ぢぃ","dʑˈiː"),
    ("づぅ","dˈɯː"),
    ("づゃ","zˈʲa"),
    ("づゅ","zˈʲɯ"),
    ("づょ","zˈʲo"),
    ("でぇ","dˈeː"),
    ("どぉ","dˈoː"),
    ("なぁ","nˈaː"),
    ("にぃ","nˈiː"),
    ("ぬぅ","nˈɯː"),
    ("ぬゃ","nˈʲa"),
    ("ぬゅ","nˈʲɯ"),
    ("ぬょ","nˈʲo"),
    ("ねぇ","nˈeː"),
    ("のぉ","nˈoː"),
    ("はぁ","hˈaː"),
    ("ひぃ","çˈiː"),
    ("ふぅ","ɸˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("へぇ","hˈeː"),
    ("ほぉ","hˈoː"),
    ("ばぁ","bˈaː"),
    ("びぃ","bˈiː"),
    ("ぶぅ","bˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ぶゅ","bˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("べぇ","bˈeː"),
    ("ぼぉ","bˈoː"),
    ("ぱぁ","pˈaː"),
    ("ぴぃ","pˈiː"),
    ("ぷぅ","pˈɯː"),
    ("ぷゃ","pˈʲa"),
    ("ぷゅ","pˈʲɯ"),
    ("ぷょ","pˈʲo"),
    ("ぺぇ","pˈeː"),
    ("ぽぉ","pˈoː"),
    ("まぁ","mˈaː"),
    ("みぃ","mˈiː"),
    ("むぅ","mˈɯː"),
    ("むゃ","mˈʲa"),
    ("むゅ","mˈʲɯ"),
    ("むょ","mˈʲo"),
    ("めぇ","mˈeː"),
    ("もぉ","mˈoː"),
    ("やぁ","jˈaː"),
    ("ゆぅ","jˈɯː"),
    ("ゆゃ","jˈaː"),
    ("ゆゅ","jˈɯː"),
    ("ゆょ","jˈoː"),
    ("よぉ","jˈoː"),
    ("らぁ","ɽˈaː"),
    ("りぃ","ɽˈiː"),
    ("るぅ","ɽˈɯː"),
    ("るゃ","ɽˈʲa"),
    ("るゅ","ɽˈʲɯ"),
    ("るょ","ɽˈʲo"),
    ("れぇ","ɽˈeː"),
    ("ろぉ","ɽˈoː"),
    ("わぁ","ɯˈaː"),
    ("をぉ","oː"),

    ("う゛","bˈɯ"),
    ("でぃ","dˈi"),
    ("でぇ","dˈeː"),
    ("でゃ","dˈʲa"),
    ("でゅ","dˈʲɯ"),
    ("でょ","dˈʲo"),
    ("てぃ","tˈi"),
    ("てぇ","tˈeː"),
    ("てゃ","tˈʲa"),
    ("てゅ","tˈʲɯ"),
    ("てょ","tˈʲo"),
    ("すぃ","sˈi"),
    ("ずぁ","zˈɯa"),
    ("ずぃ","zˈi"),
    ("ずぅ","zˈɯ"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ずぇ","zˈe"),
    ("ずぉ","zˈo"),
    ("きゃ","kˈʲa"),
    ("きゅ","kˈʲɯ"),
    ("きょ","kˈʲo"),
    ("しゃ","ɕˈʲa"),
    ("しゅ","ɕˈʲɯ"),
    ("しぇ","ɕˈʲe"),
    ("しょ","ɕˈʲo"),
    ("ちゃ","tɕˈa"),
    ("ちゅ","tɕˈɯ"),
    ("ちぇ","tɕˈe"),
    ("ちょ","tɕˈo"),
    ("とぅ","tˈɯ"),
    ("とゃ","tˈʲa"),
    ("とゅ","tˈʲɯ"),
    ("とょ","tˈʲo"),
    ("どぁ","dˈoa"),
    ("どぅ","dˈɯ"),
    ("どゃ","dˈʲa"),
    ("どゅ","dˈʲɯ"),
    ("どょ","dˈʲo"),
    ("どぉ","dˈoː"),
    ("にゃ","nˈʲa"),
    ("にゅ","nˈʲɯ"),
    ("にょ","nˈʲo"),
    ("ひゃ","çˈʲa"),
    ("ひゅ","çˈʲɯ"),
    ("ひょ","çˈʲo"),
    ("みゃ","mˈʲa"),
    ("みゅ","mˈʲɯ"),
    ("みょ","mˈʲo"),
    ("りゃ","ɽˈʲa"),
    ("りぇ","ɽˈʲe"),
    ("りゅ","ɽˈʲɯ"),
    ("りょ","ɽˈʲo"),
    ("ぎゃ","gˈʲa"),
    ("ぎゅ","gˈʲɯ"),
    ("ぎょ","gˈʲo"),
    ("ぢぇ","dʑˈe"),
    ("ぢゃ","dʑˈa"),
    ("ぢゅ","dʑˈɯ"),
    ("ぢょ","dʑˈo"),
    ("じぇ","dʑˈe"),
    ("じゃ","dʑˈa"),
    ("じゅ","dʑˈɯ"),
    ("じょ","dʑˈo"),
    ("びゃ","bˈʲa"),
    ("びゅ","bˈʲɯ"),
    ("びょ","bˈʲo"),
    ("ぴゃ","pˈʲa"),
    ("ぴゅ","pˈʲɯ"),
    ("ぴょ","pˈʲo"),
    ("うぁ","ɯˈa"),
    ("うぃ","ɯˈi"),
    ("うぇ","ɯˈe"),
    ("うぉ","ɯˈo"),
    ("うゃ","ɯˈʲa"),
    ("うゅ","ɯˈʲɯ"),
    ("うょ","ɯˈʲo"),
    ("ふぁ","ɸˈa"),
    ("ふぃ","ɸˈi"),
    ("ふぅ","ɸˈɯ"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("ふぇ","ɸˈe"),
    ("ふぉ","ɸˈo"),

    ("あ","a"),
    ("い","i"),
    ("う","ɯ"),
    ("え","e"),
    ("お","o"),
    ("か","kˈa"),
    ("き","kˈi"),
    ("く","kˈɯ"),
    ("け","kˈe"),
    ("こ","kˈo"),
    ("さ","sˈa"),
    ("し","ɕˈi"),
    ("す","sˈɯ"),
    ("せ","sˈe"),
    ("そ","sˈo"),
    ("た","tˈa"),
    ("ち","tɕˈi"),
    ("つ","tsˈɯ"),
    ("て","tˈe"),
    ("と","tˈo"),
    ("な","nˈa"),
    ("に","nˈi"),
    ("ぬ","nˈɯ"),
    ("ね","nˈe"),
    ("の","nˈo"),
    ("は","hˈa"),
    ("ひ","çˈi"),
    ("ふ","ɸˈɯ"),
    ("へ","hˈe"),
    ("ほ","hˈo"),
    ("ま","mˈa"),
    ("み","mˈi"),
    ("む","mˈɯ"),
    ("め","mˈe"),
    ("も","mˈo"),
    ("ら","ɽˈa"),
    ("り","ɽˈi"),
    ("る","ɽˈɯ"),
    ("れ","ɽˈe"),
    ("ろ","ɽˈo"),
    ("が","gˈa"),
    ("ぎ","gˈi"),
    ("ぐ","gˈɯ"),
    ("げ","gˈe"),
    ("ご","gˈo"),
    ("ざ","zˈa"),
    ("じ","dʑˈi"),
    ("ず","zˈɯ"),
    ("ぜ","zˈe"),
    ("ぞ","zˈo"),
    ("だ","dˈa"),
    ("ぢ","dʑˈi"),
    ("づ","zˈɯ"),
    ("で","dˈe"),
    ("ど","dˈo"),
    ("ば","bˈa"),
    ("び","bˈi"),
    ("ぶ","bˈɯ"),
    ("べ","bˈe"),
    ("ぼ","bˈo"),
    ("ぱ","pˈa"),
    ("ぴ","pˈi"),
    ("ぷ","pˈɯ"),
    ("ぺ","pˈe"),
    ("ぽ","pˈo"),
    ("や","jˈa"),
    ("ゆ","jˈɯ"),
    ("よ","jˈo"),
    ("わ","ɯˈa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("ん","ɴ"),
    ("っ","ʔ"),
    ("ー","ː"),

    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","ɯ"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","ɯˈa"),
    ("ぉ","o"),

    ("を","o")
])

nasal_sound = OrderedDict([
    # before m, p, b
    ("ɴm","mm"),
    ("ɴb", "mb"),
    ("ɴp", "mp"),
    
    # before k, g
    ("ɴk","ŋk"),
    ("ɴg", "ŋg"),
    
    # before t, d, n, s, z, ɽ
    ("ɴt","nt"),
    ("ɴd", "nd"),
    ("ɴn","nn"),
    ("ɴs", "ns"),
    ("ɴz","nz"),
    ("ɴɽ", "nɽ"),
    
    ("ɴɲ", "ɲɲ"),
    
])

def hiragana2IPA(text):
    orig = text

    for k, v in kana_mapper.items():
        text = text.replace(k, v)

    for k, v in nasal_sound.items():
        text = text.replace(k, v)
        
    return text

def alphabetreading(text):
    alphabet_dict = {"A": "エイ",
                "B": "ビー",
                 "C": "シー",
                 "D": "ディー",
                 "E": "イー",
                 "F": "エフ",
                 "G": "ジー",
                 "H": "エイチ",
                "I":"アイ",
                "J":"ジェイ",
                "K":"ケイ",
                "L":"エル",
                "M":"エム",
                "N":"エヌ",
                "O":"オー",
                "P":"ピー",
                "Q":"キュー",
                "R":"アール",
                "S":"エス",
                "T":"ティー",
                "U":"ユー",
                "V":"ヴィー",
                "W":"ダブリュー",
                "X":"エックス",
                "Y":"ワイ",
                "Z":"ゼッド"}
    text = text.upper()
    text_ret = ""
    for t in text:
        if t in alphabet_dict:
            text_ret += alphabet_dict[t]
        else:
            text_ret += t
    return text_ret

def extract_features(text, tokenizer):
    results = pyopenjtalk.run_frontend(unicodedata.normalize('NFKC', text).replace('\n', ''))
    
    words = []

    for n in results:
        tokens = unicodedata.normalize("NFKC", n["string"])  # Lấy từ gốc
        kana = unicodedata.normalize("NFKC", n["read"])  # Phiên âm Katakana

        if n["pos"] == "記号":  # Nếu là ký hiệu, dấu câu
            words.append({
                'tokens': tokenizer.encode([tokens])[1],  
                'words': tokens,  
                'kana': alphabetreading(tokens)  # Chuyển ký hiệu thành dạng đọc nếu cần
            })
        else:  # Xử lý các từ khác
            if kana in ["ュ", "ャ", "ョ"]:  # Nếu là âm nhỏ, ghép với từ trước đó
                if words:  # Kiểm tra tránh lỗi danh sách rỗng
                    words[-1]['words'] += tokens
                    words[-1]['kana'] += kana
            else:
                words.append({
                    'tokens': tokenizer.encode([tokens])[1],  
                    'words': tokens,  
                    'kana': kana
                })

    return words

def phonemize(text, tokenizer):
    texts = text.split('。')
    words = []
    for text in texts:
        if text == "":
            continue
        if text[-1] != "。":
            text += "。"
        words.extend(extract_features(text, tokenizer))
    input_ids = [w['tokens'] for w in words]
    phonemes = [w['kana'] for w in words]
    
    return {'input_ids' : input_ids, 'phonemes': phonemes} 