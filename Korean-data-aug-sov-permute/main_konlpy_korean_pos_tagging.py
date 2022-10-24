from konlpy import tag

"""https://github.com/konlpy/konlpy/blob/master/docs/morph.py"""


def tagging(tagger, text):
    ret_list = []
    try:
        ret_list = getattr(tag, tagger)().pos(text)
    except Exception as e:
        print("Uhoh,", e)

    return ret_list


if __name__ == '__main__':
    examples = [
        u'안녕하세요 저는 사과를 좋아하는 홍길동입니다',  # 일반 문장
        u'철수가 꽃을 영희에게 준다',
        # u'아버지가방에들어가신다',  # 띄어 쓰기
        # u'나는 밥을 먹는다',  # 중의성 해소
        # u'하늘을 나는 자동차',
        # u'아이폰 기다리다 지쳐 애플공홈에서 언락폰질러버렸다 6+ 128기가실버ㅋ',  # 속어
    ]

    taggers = [t for t in dir(tag) if t[0].isupper()]

    for tagger in taggers:
        if tagger == 'Twitter':
            continue

        for example in examples:
            print(f'{tagger}: {tagging(tagger, example)}')
