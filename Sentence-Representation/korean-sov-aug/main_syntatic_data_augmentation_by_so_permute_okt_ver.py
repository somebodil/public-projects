import copy
import random

import torch
from konlpy.tag import Okt
from transformers import ElectraTokenizer, ElectraForPreTraining

okt = Okt()
okt_josa = 'Josa'
okt_verb = 'Verb'

pretrained_model_name_or_path = "monologg/koelectra-base-v3-discriminator"


def sov_parse(sentence):
    noun_josa = []
    verb = copy.deepcopy(sentence)

    pos_tag = okt.pos(sentence)
    max_tag_idx = len(pos_tag) - 1
    for i, (word, tag) in enumerate(pos_tag):
        if tag == okt_josa and i != max_tag_idx - 1 and i != max_tag_idx:
            wi = verb.find(word)
            wi += len(word)
            noun_josa.append(verb[:wi].strip())
            verb = verb[wi:].strip()

    return noun_josa, [verb]


tokenizer = ElectraTokenizer.from_pretrained(pretrained_model_name_or_path)
discriminator = ElectraForPreTraining.from_pretrained(pretrained_model_name_or_path)


def is_all_ok(sentence):
    example = tokenizer(sentence, return_tensors='pt')
    predictions = discriminator(input_ids=example['input_ids'], attention_mask=example['attention_mask'])
    predictions = torch.round((torch.sign(predictions['logits']) + 1) / 2)
    return not predictions.squeeze().detach().numpy().any(0)


def print_permuted_example(sentence):
    noun_josa, verb = sov_parse(sentence)

    print('----------------------------------------------------')
    print(f"original: {sentence}")

    for i in range(30):
        random.shuffle(noun_josa)
        if is_all_ok((noun_josa + verb).__str__()):
            print(f"positive: {noun_josa + verb}")


def main():
    print_permuted_example(u'뉴욕에서 철수는 식당에서 피자를 먹는 민준이를 만났다.')


if __name__ == '__main__':
    main()
