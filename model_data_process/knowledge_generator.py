from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

from utils.comet import Comet

nltk.download('popular')


class KnowledgeGenerator:
    # todo: device based on device
    COMET = Comet(model_path='smetan/comet-bart-aaai', device='')
    STOP_WORDS = set(stopwords.words('english'))
    SOCIAL_INTERACTION_REL = ['xIntent', 'xReact', 'xNeed', 'xWant', 'xEffect']
    EVENT_CENTERED_REL = ['Causes', 'HinderedBy', 'xReason', 'isAfter', 'isBefore', 'HasSubevent', 'isFilledBy']
    PHYSICAL_ENTITIES = ['ObjectUse', 'CapableOf', 'HasProperty', 'Desires', ]

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
            if gen_result.lower() not in ['none', 'null', ]:
                cleaned_results.append(cleaned_gen_result)
        return list(set(cleaned_results))

    @classmethod
    def get_social_interaction_knowledge(cls, text: str) -> dict:
        """
        get social interactions knowledge by generating nodes
        :param text:
        :return:
        """
        return cls._generate_knowledge(texts=[text, ], relations=cls.SOCIAL_INTERACTION_REL)

    @classmethod
    def get_event_base_knowledge(cls, text: str) -> dict:
        """
        generate some nodes that have event-based relations with text
        :param text: 
        :return: 
        """
        tokenized_sentences = sent_tokenize(text)
        return cls._generate_knowledge(texts=tokenized_sentences, relations=cls.EVENT_CENTERED_REL)

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
            sentence_entity = [each[0] for each in tagged if 'NN' in each[1] and (tagged not in all_entities)]
            all_entities += sentence_entity

        # generate nodes for each entity and rel
        return cls._generate_knowledge(texts=all_entities, relations=cls.PHYSICAL_ENTITIES)

    @classmethod
    def _generate_knowledge(cls, texts: list, relations: list):
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
        for entity in texts:
            gen_results = dict()
            for rel in relations:
                generated_nodes = cls.clean_comet_results(cls.COMET.generate(entity, rel=rel))
                gen_results.update({rel: generated_nodes})
            result.update({entity: gen_results})

        return result
    
    @classmethod
    def __call__(cls, texts: list, get_all_knw=True, *args, **kwargs):
        # todo: complete
        if len(texts) == 0:
            return None
        last_utter = texts[-1]
        cls.get_social_interaction_knowledge(text=last_utter)
        cls.get_event_base_knowledge(text=". ".join(texts))
        cls.get_entity_knowledge(text=", ".join(texts))


