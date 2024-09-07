from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import torch

from utils.comet import Comet

nltk.download('popular')


class KnowledgeGenerator:
    COMET = Comet(model_path='smetan/comet-bart-aaai', device='cuda' if torch.cuda.is_available() else 'cpu')
    STOP_WORDS = set(stopwords.words('english'))
    SOCIAL_INTERACTION_REL = {'xIntent': 'because X wanted',
                              'xReact': 'as a result, X feels',
                              'xNeed': 'but before, X needed',
                              'xWant': 'as a result, X wants',
                              'xEffect': 'as a result, X wil', }
    EVENT_CENTERED_REL = {'Causes': 'causes',
                          'HinderedBy': 'can be hindered by',
                          'xReason': 'because',
                          'isAfter': 'happens after',
                          'isBefore': 'happens before',
                          'HasSubEvent': 'includes the event/action',
                          'isFilledBy': 'blank can be filled by', }
    PHYSICAL_ENTITIES = {'ObjectUse': 'is used for',
                         'CapableOf': 'is/are capable of',
                         'HasProperty': 'can be characterized by being/having',
                         'Desires': 'desires', }

    @staticmethod
    def clean_comet_results(results: list) -> list:
        """
        clean the result of comet:
        1. delete extra space in text
        2. delete the generated results if it is none or null
        :param results: comet result
        :return:
        """
        cleaned_results = list()
        for gen_result in results:
            cleaned_gen_result = " ".join(gen_result.split())
            if cleaned_gen_result.lower() not in ['none', 'null', ]:
                cleaned_results.append(cleaned_gen_result)
        return list(set(cleaned_results))

    @classmethod
    def get_social_interaction_knowledge(cls, text: str) -> dict:
        """
        get social interactions knowledge by generating nodes
        :param text:
        :return:
        """

        return cls._generate_knowledge(texts=[text for i in range(len(cls.SOCIAL_INTERACTION_REL.keys()))],
                                       relations=list(cls.SOCIAL_INTERACTION_REL.keys()))

    @classmethod
    def get_event_base_knowledge(cls, text: str) -> dict:
        """
        generate some nodes that have event-based relations with text
        :param text: 
        :return:
        """
        tokenized_sentences = sent_tokenize(text)
        text_list = list()
        relations = list()
        for txt in tokenized_sentences:
            if txt not in text_list:
                for rel in list(cls.EVENT_CENTERED_REL.keys()):
                    text_list.append(txt)
                    relations.append(rel)

        return cls._generate_knowledge(texts=text_list, relations=relations)

    @classmethod
    def get_entity_knowledge(cls, text: str) -> dict:
        """
        generate some nodes that have entity-based relations with entities of text
        :param text: 
        :return:
        """
        all_entities = list()

        # get all entities
        tokenized_sentences = sent_tokenize(text)
        for each_sentence in tokenized_sentences:
            words_list = nltk.word_tokenize(each_sentence)
            words_list = [word for word in words_list if word not in cls.STOP_WORDS]
            # apply part of speech
            tagged = nltk.pos_tag(words_list)
            # get all nouns as entities
            # set isn't used for remain seq sort on text
            sentence_entity = [each[0] for each in tagged if 'NN' in each[1] and (each[0] not in all_entities)]
            all_entities += sentence_entity

        text_list = list()
        relations = list()
        for txt in all_entities:
            if txt not in text_list:
                for rel in list(cls.PHYSICAL_ENTITIES.keys()):
                    text_list.append(txt)
                    relations.append(rel)

        # generate nodes for each entity and rel
        return cls._generate_knowledge(texts=text_list, relations=relations)

    @classmethod
    def _generate_knowledge(cls, texts: list, relations: list, batch_size=32):
        """
        generate nodes for each nodes and relation
        :param texts: list of nodes
        :param relations: list of relations
        :return: dictionary with below format
            {
                text_1: {
                    rel_1: [node1, node2, ...],
                    rel_2: [node1, node2, ...],
                },
                text_2: {
                    rel_1: [node1, node2, ...],
                    rel_2: [node1, node2, ...],
                }
            }
        """
        result = dict()
        for i in range(int(len(texts) / batch_size) + 1):
            batched_text = texts[i * batch_size: (i + 1) * batch_size]
            batch_rel = relations[i * batch_size: (i + 1) * batch_size]
            batch_generated_nodes = cls.COMET.generate(batched_text, rel=batch_rel)
            generated_nodes = [cls.clean_comet_results(each) for each in batch_generated_nodes]

            # aggregate result
            for i in range(len(batched_text)):
                k = batched_text[i]
                v = {batch_rel[i]: generated_nodes[i]}
                value_total_result = result.get(k, dict())
                value_total_result.update(v)
                result.update({k: value_total_result})

        return result

        # result = dict()
        # for entity in texts:
        #     gen_results = dict()
        #     for rel in relations:
        #         generated_nodes = cls.clean_comet_results(cls.COMET.generate(entity, rel=rel))
        #         gen_results.update({rel: generated_nodes})
        #     result.update({entity: gen_results})
        #
        # return result
    
    @classmethod
    def run(cls, texts: list, get_all_knw: bool = True):
        """
        generate data for a list of text
        :param texts: all utterance
        :param get_all_knw: a boolean shows that just get social or all knowledge
        :return:
        """

        if len(texts) == 0:
            return dict()
        last_utter = texts[-1]
        social_result = cls.get_social_interaction_knowledge(text=last_utter)
        if not get_all_knw:
            return social_result, None, None
        return (social_result,
                cls.get_event_base_knowledge(text=". ".join([text for i, text in enumerate(texts) if i % 2 == 0]).replace('..', '.')),
                cls.get_entity_knowledge(text=". ".join([text for i, text in enumerate(texts) if i % 2 == 0]).replace('..', '.')))
