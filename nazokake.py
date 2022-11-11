# -*- coding: utf-8 -*-
from operator import truediv
import gensim
import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM
from pprint import pprint
import sqlite3
import sys
import datetime
import random
import numpy as np
import math

print(datetime.datetime.now())

# 同音異義語
dbname = "homonym.db"
## chiVeデータのPATH（kv:KeyedVectors）
model_path = "/data/nazokake/chive/chive-1.2-mc30_gensim/chive-1.2-mc30.kv"
wv = gensim.models.KeyedVectors.load(model_path)
#FastText
#model_path = "/data/nazokake/fasttextmodel/cc.ja.300.vec.gz"
#wv =  gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
bert_path = "/data/nazokake/bert"
# bertデータ(初回だけダウンロードに時間かかる)
# tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_path)
# print("BertJapaneseTokenizer.from_pretrained done")
# Load pre-trained model
# model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
model = BertForMaskedLM.from_pretrained(bert_path)
# print(datetime.datetime.now())
model.eval()
print("model load done")
print(datetime.datetime.now())

masked_buns = []
masked_buns.append({"bun": "つまりXXXにYYYは必要です", "gobi": "がつきものです"})
masked_buns.append({"bun": "つまりXXXはYYYに必要です", "gobi": "に欠かせません"})
masked_buns.append({"bun": "つまりXXXはYYYが大事です", "gobi": "が欠かせません"})
masked_buns.append({"bun": "さらにXXXはYYYが課題です。", "gobi": "がつきものです"})


def check_by_word(conn, odai_word, word):
    if odai_word in word or word in odai_word:
        return False
    else:
        cur = conn.cursor()
        cur.execute("select pron1 from homonyms where word = ?", (word,))
        rows = cur.fetchall()
        if len(rows) >= 1:
            return rows[0][0]
        else:
            return False


def get_same_pron_word_list(conn, odai_word, word, pron):
    cur = conn.cursor()
    cur.execute(
        "select word,pron1 from homonyms where pron1 = ? and word <> ? ", (pron, word)
    )
    rows = cur.fetchall()
    same_pron_word_list = []
    if len(rows) >= 1:
        for row in rows:
            same_pron_word_list.append({"word": row[0]})
    cur.close()
    return same_pron_word_list


def get_relation_word_list(conn, text, top_rank):
    tokenized_text = tokenizer.tokenize(text)
    masked_index = 0
    for idx, token_word in enumerate(tokenized_text):
        if token_word == "[MASK]":
            masked_index = idx
    # pprint(tokenized_text)
    # pprint(masked_index)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        # pprint(outputs)
        predictions = outputs[0][0, masked_index].topk(top_rank)
    # print(predictions)
    relation_word_list = []
    for index_t in predictions.indices:
        index = index_t.item()
        word = tokenizer.convert_ids_to_tokens([index])[0]
        if len(word) > 1:
            relation_word_list.append(word)
    relation_word_score_list = []
    for index_t in predictions.values:
        score = index_t.item()
        relation_word_score_list.append(score)
    return relation_word_list, relation_word_score_list


def get_relation_word_yomi_list(
    conn, odai_word, relation_word_list, relation_word_score_list
):
    relation_word_yomi_list = []
    for idx, relation_word_score in enumerate(relation_word_list):
        pron = check_by_word(conn, odai_word, relation_word_score[0])
        if pron:
            relation_word_yomi_list.append(
                {
                    "word": relation_word_score[0],
                    "yomi": pron,
                    "score": relation_word_score[1],
                    "bun_score": relation_word_score_list[idx],
                }
            )
    return relation_word_yomi_list


def get_relation_word_list_by_word2vec(
    odai_word, relation_word_list, relation_word_score_list, similar_word_distance_list
):

    word_list = []
    word_score_list = []
    for idx, relation_word in enumerate(relation_word_list):
        for similar_word_distance in similar_word_distance_list:
            if similar_word_distance[0] == relation_word:
                word_list.append(similar_word_distance)
                word_score_list.append(relation_word_score_list[idx])
    return word_list, word_score_list


def get_drafts_by_answer_word_list(
    odai_word,
    same_pron_word,
    gobi,
    relation_word_yomi,
    answer_word_list,
    answer_word_score_list,
    loop_break_count,
    similar_word_distance_list_buffers,
):
    if same_pron_word["word"] in similar_word_distance_list_buffers.keys():
        similar_word_distance_list = similar_word_distance_list_buffers[
            same_pron_word["word"]
        ]
    else:
        try:
            similar_word_distance_list = wv.most_similar(
                [same_pron_word["word"]], negative=[odai_word], topn=600
            )
            similar_word_distance_list_buffers[
                same_pron_word["word"]
            ] = similar_word_distance_list
        except:
            return loop_break_count, []
    drafts_by_answer_word_list = []
    for idx, answer_word in enumerate(answer_word_list):
        for similar_word_distance in similar_word_distance_list:
            if similar_word_distance[0] == answer_word:
                drafts_by_answer_word_list.append(
                    {
                        "answer_word": answer_word,
                        "relation_word": relation_word_yomi["word"],
                        "same_pron_word": same_pron_word["word"],
                        "yomi": relation_word_yomi["yomi"],
                        "relation_word_score": relation_word_yomi["score"],
                        "relation_bun_score": relation_word_yomi["bun_score"],
                        "similar_word_score": similar_word_distance[1],
                        "similar_bun_score": answer_word_score_list[idx],
                        "gobi": gobi,
                    }
                )
                loop_break_count += 1
    return loop_break_count, drafts_by_answer_word_list


def get_drafts_for_add(
    odai_word, bun, gobi, answer, conn, similar_word_distance_list_buffers
):
    loop_break_count = 0
    drafts_for_add = []
    for relation_word_yomi in answer["relation_word_yomi_list"]:
        if loop_break_count > 40:
            break
        for same_pron_word in relation_word_yomi["same_pron_word_list"]:
            kokoro_sentence = bun.replace("YYY", same_pron_word["word"]).replace(
                "XXX", "[MASK]"
            )
            answer_word_list, answer_word_score_list = get_relation_word_list(
                conn, kokoro_sentence, 300
            )
            (
                loop_break_count,
                drafts_by_answer_word_list,
            ) = get_drafts_by_answer_word_list(
                odai_word,
                same_pron_word,
                gobi,
                relation_word_yomi,
                answer_word_list,
                answer_word_score_list,
                loop_break_count,
                similar_word_distance_list_buffers,
            )
            drafts_for_add.extend(drafts_by_answer_word_list)

    return drafts_for_add


def get_best_draft(drafts):
    if len(drafts) == 0:
        return {
            "answer_word": "リンゴ",
            "relation_word": "気",
            "same_pron_word": "木",
            "yomi": "キ",
            "relation_word_score": 0.0,
            "relation_bun_score": 0.0,
            "similar_word_score": 0.0,
            "similar_bun_score": 0.0,
            "gobi": "になるでしょう",
        }
    drafts_sorted1 = sorted(
        drafts, key=lambda draft: draft["similar_bun_score"], reverse=True
    )
    drafts_cut1 = drafts_sorted1[0 : int(math.ceil(len(drafts_sorted1) / 2))]
    drafts_sorted2 = sorted(
        drafts_cut1,
        key=lambda drafts_cut1: drafts_cut1["relation_bun_score"],
        reverse=True,
    )
    drafts_cut2 = drafts_sorted2[0 : int(math.ceil(len(drafts_sorted2) / 2))]
    drafts_sorted3 = sorted(
        drafts_cut2,
        key=lambda drafts_cut2: drafts_cut2["relation_word_score"],
        reverse=True,
    )
    drafts_cut3 = drafts_sorted3[0 : int(math.ceil(len(drafts_sorted3) / 2))]
    drafts_sorted4 = sorted(
        drafts_cut2,
        key=lambda drafts_cut3: drafts_cut3["similar_word_score"],
        reverse=True,
    )
    drafts_cut4 = drafts_sorted4[0 : int(math.ceil(len(drafts_sorted4) / 2))]
    return drafts_cut4[0]


def print_draft(odai_word, draft, debug):
    print("「" + odai_word + "」とかけて「" + draft["answer_word"] + "」と説く　その心は!")
    if debug is False:
        input("push return key ")
    print(
        "どちらも「"
        + draft["relation_word"]
        + "/"
        + draft["same_pron_word"]
        + "("
        + draft["yomi"].strip()
        + ")」"
        + draft["gobi"]
    )
    if debug:
        print(
            "relation_word_score:"
            + str(draft["relation_word_score"])
            + " similar_word_score:"
            + str(draft["similar_word_score"])
        )
        print(
            "relation_bun_score:"
            + str(draft["relation_bun_score"])
            + " similar_bun_score:"
            + str(draft["similar_bun_score"])
        )


def main_func(odai_word, debug):
    conn = sqlite3.connect(dbname)
    conn.text_factory = str
    rng = np.random.default_rng()
    rng.shuffle(masked_buns, axis=0)
    drafts = []
    similar_word_distance_list_buffers = {}
    try:
        similar_word_distance_list = wv.most_similar([odai_word], topn=500)
    except:
        similar_word_distance_list = []
    for masked_bun in masked_buns:
        answer = {}
        answer["masked_bun"] = masked_bun
        odai_sentence = (
            masked_bun["bun"].replace("XXX", odai_word).replace("YYY", "[MASK]")
        )
        relation_word_list, relation_word_score_list = get_relation_word_list(
            conn, odai_sentence, 500
        )
        # word2vecでしぼりこみ
        (
            answer["relation_word_list"],
            answer["relation_word_score_list"],
        ) = get_relation_word_list_by_word2vec(
            odai_word,
            relation_word_list,
            relation_word_score_list,
            similar_word_distance_list,
        )
        answer["relation_word_yomi_list"] = get_relation_word_yomi_list(
            conn,
            odai_word,
            answer["relation_word_list"],
            answer["relation_word_score_list"],
        )
        for idx, relation_word_yomi in enumerate(answer["relation_word_yomi_list"]):
            answer["relation_word_yomi_list"][idx][
                "same_pron_word_list"
            ] = get_same_pron_word_list(
                conn, odai_word, relation_word_yomi["word"], relation_word_yomi["yomi"]
            )
        drafts.extend(
            get_drafts_for_add(
                odai_word,
                masked_bun["bun"],
                masked_bun["gobi"],
                answer,
                conn,
                similar_word_distance_list_buffers,
            )
        )
    if debug:
        for draft in drafts:
            print_draft(odai_word, draft, debug)
    best_draft = get_best_draft(drafts)
    print(datetime.datetime.now())
    print_draft(odai_word, best_draft, debug)


if __name__ == "__main__":
    print("終了したい場合はexit")
    while True:
        print("")

        inputs = input("お題を入力してください: ").split(" ")
        odai_word = inputs[0]
        if odai_word == "exit":
            break
        if odai_word == "":
            continue
        if len(inputs) > 1 and inputs[1] == "debug":
            debug = True
        else:
            debug = False
        print(datetime.datetime.now())
        main_func(odai_word, debug)
