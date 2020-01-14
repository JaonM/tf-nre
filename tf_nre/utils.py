import re


def parse_text(text):
    """
    处理文本，返回主体客体token位置以及纯文本
    :param text:
    :return: e1 pos,e2pos,clean text
    """
    e1_pos, e2_pos = 0, 0
    tokens = text.split()
    e1_pat = re.compile(r'<e1>.*</e1>')
    e2_pat = re.compile(r'<e2>.*</e2>')
    for i in range(len(tokens)):
        if e1_pat.match(tokens[i]):
            e1_pos = i
            tokens[i] = re.sub('<e1>|</e1>', '', tokens[i])
        if e2_pat.match(tokens[i]):
            e2_pos = i
            tokens[i] = re.sub('<e2>|</e2>', '', tokens[i])
    return e1_pos, e2_pos, ' '.join(tokens)
