import json
import random
import argparse
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

#TODO: clean up old prompts
    # 'basic':"""
    #     You are a carefull medical expert in writing you consider background knowledge 
    #     and medical experties when dealing with tasks you solve. You think about
    #     the task and how to output and produce real example of text simple sentence without decoration. 
    #     """,
        # 'kshot_num_sent_entity': """Generate {number_of_sentences} sentences that contain the disease or diagnosis {condition}, as it would appear in a {genre}.
        # Requirements:
        # Keep the sentences factual, and contextually realistic for clinical or biomedical text.
        # Output only sentences — no explanations or extra text or decorations. 
        # Use the following examples as inspiration for style and structure from provided part of speech tags and dependency parsing tags.
        # Examples:
        # {text}
        # Your task is to produce just the sentences, given the upper requirements.
        # Sentences:
        # """,
        #     'role':"""
        # You are a careful {role} with medical expertise. 
        # Output raw plain text only. No formatting, emojis, or commentary. 

        # """,

    # 'syn_generation': """
    #     A diagnosis is the process of identifying a specific disease or condition 
    #     that is dependent of patient's signs, symptoms, and medical history. 
    #     A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
    #     structural or functional state of an organism, associated with specific signs and symptoms and 
    #     distinct from physical injury. It can be caused by various factors, including pathogens, 
    #     genetic or autoimmune dysfunction, or environmental influences.
    #     Diseases are generally characterized by a pathological process and 
    #     a specific set of symptoms, diseases can be grouped in Abnormal Condition, 
    #     Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
    #     Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, 
    #     Autoimmune Diseases, Degenerative Diseases.
    #     For given <disease> think about context in what kind of sentence could it occure
    #     your task is to generate just one a sentence containing the given <disease>, 
    #     you don't decorate or generate anything else. 
    #     You create a single sentence of text with the term used as it would be used in 
    #     medical document. Remember keep it simple. 
    #     DO not use word patient or individual or subject in your generated text. 
    #     Consider somethig you would see in an anamnesis or abstract or case-study or some other 
    #     medical document. Consider the sentence output that comes after sentence: 
    #     The patient diagnosed with {condition}.        
    #     Based on your medical expertise. 
    #     Your task is to generate next sentence containing following disease: {condition}. \n\n""",

    # 'genre_syn_generation': """
    #     A diagnosis is the process of identifying a specific disease or condition 
    #     that is dependent of patient's signs, symptoms, and medical history. 
    #     A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
    #     structural or functional state of an organism, associated with specific signs and symptoms and 
    #     distinct from physical injury. It can be caused by various factors, including pathogens, 
    #     genetic or autoimmune dysfunction, or environmental influences.
    #     Diseases are generally characterized by a pathological process and 
    #     a specific set of symptoms, diseases can be grouped in Abnormal Condition, 
    #     Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
    #     Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, 
    #     Autoimmune Diseases, Degenerative Diseases.
    #     For given <disease or diagnosis> think about context in what kind of sentence could it occure
    #     your task is to generate just one a sentence containing the given {condition}, 
    #     you don't decorate or generate anything else. 
    #     You create a single sentence of text with the entity {condition} used as it would be used in 
    #     medical document. Remember keep it simple. Do not use word patient or individual or subject in your generated text. 
    #     Consider somethig you would see in an {genre}. Consider the sentence output that comes after sentence: 
    #     The patient diagnosed with {condition}. 
    #     Based on your medical expertise. 
    #     Your task is to generate next sentence containing following disease or diagnosis <{condition}>. \n\n""",
            # genre = anamnesis, abstract, case-study
        # We want to check if there are additional diagnoses in produced sentence 
        # ('this is the part where there could be noise)
        # json output
        
        # Generate one sentence that naturally includes the disease or diagnosis {condition}, as it would appear in a {genre}.
        # Requirements:
        # Use {condition} exactly as written within the sentence.
        # Keep the sentence factual, and contextually realistic for clinical or biomedical text.
        # Output only the sentence — no explanations or extra text.
        # Use the following examples as inspiration for linguistic style and structure from provided sentence, part of speech tags (POS) and dependency parsing tags (DEP).
    #         'disease_annotation': """Find and extract diagnoses and diseases from the text and. 
    #     Your output should be a list of entities that is machine-readable using eval(): 
    #     ["entity_text", "entity_text", ...]. 
    #     Think about context of the text. As a medical expert, you know that patient symptoms, 
    #     expert-defined signs, diagnostic and medical tests, and treatments are NOT diagnoses or diseases. 
    #     You search for diagnoses and disease: 
    #     A diagnosis is the process of identifying a specific disease or condition 
    #     that is dependent of patient's signs, symptoms, and medical history. 
    #     A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
    #     structural or functional state of an organism, associated with specific signs and symptoms and 
    #     distinct from physical injury. It can be caused by various factors, including pathogens, 
    #     genetic or autoimmune dysfunction, or environmental influences.
    #     Diseases are generally characterized by a pathological process and a specific set of symptoms, diseases can be 
    #     grouped in Abnormal Condition, Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
    #     Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, Autoimmune Diseases, Degenerative Diseases.
    #     Extract only diagnoses and dieases string sturctured ["entity_text", "entity_text", ...]
    #     if you don't find any entites output an empty list, i.e. []. Some form of {condition} is in the sentence.
    #     Diseases or diagnoses ["entity_text","entity_text",...] must be the exact copy of spans from the sentence: 
    #     {text} \n\n""",    
    # 'disease_annotation_kshot': """Find and extract diagnoses and diseases from the text and. 
    #     Your output should be a list of entities that is machine-readable using eval(): 
    #     ["entity_text", "entity_text", ...]. 
    #     Think about context of the text. As a medical expert, you know that patient symptoms, 
    #     expert-defined signs, diagnostic and medical tests, and treatments are NOT diagnoses or diseases. 
    #     You search for diagnoses and disease: 
    #     A diagnosis is the process of identifying a specific disease or condition 
    #     that is dependent of patient's signs, symptoms, and medical history. 
    #     A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
    #     structural or functional state of an organism, associated with specific signs and symptoms and 
    #     distinct from physical injury. It can be caused by various factors, including pathogens, 
    #     genetic or autoimmune dysfunction, or environmental influences.
    #     Diseases are generally characterized by a pathological process and a specific set of symptoms, diseases can be 
    #     grouped in Abnormal Condition, Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
    #     Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, Autoimmune Diseases, Degenerative Diseases.
    #     Extract only diagnoses and dieases string sturctured ["entity_text", "entity_text", ...]
    #     if you don't find any entites output an empty list, i.e. []. 
    #     ##Examples:
    #     {kshot_text}
    #     ##Task:
    #     Some form of {condition} is in the sentence.
    #     Diseases or diagnoses ["entity_text","entity_text",...] must be the exact copy of spans. 
    #     Sentence:
    #     {text}
    #     Entities:""",
        
SYSTEM_PROMPTS = {

    'role_sent_type':"""
        You are a careful {language} {role} with medical expertise. Generate exactly one linguistically {sent_type} sentence in {language}. 
        Output {language} raw plain text only. no formatting, no emojis, no commentary, no markdown, no blocks.
        """,
    'annotation':"""You are a carefull medical expert in finding medical entities in text. You take context in to account when searching for entities and output the desired structure as a list ["entity_text", "entity_text", ...]. 
        "entity_text" part of the sentence is the exact copy from the given sentence to analyse. If you find <entity_text> span that has a word within the main entity, you should transcribe the whole "entity_text" as written in the sentence you are analysing. 
        You make sure you annotate only the correct entites. You need to carefully consider the entity order in sentence.
        Annotation guidelines:
            1. Annotate terms individually: Example: “Huntington’s disease (HD)” → the annotations “Huntington’s disease” and “HD” should be separate. Example: In the phrase “tense and worried”, since each word is a symptom on its own, annotate “tense” separately and “worried” separately.
            2. When annotating, if the text contains: "the patient feels pain in the fingers last ..." do not annotate it is not a disease or diagnosis.
                "Cystic fibrosis is a disease that ..." annotate as ["cystic fibrosis",] Modified entities and terms are annotated and transcribed exactly as they appear in the text; their form is not changed or reduced to the infinitive.
            3. Do not annotate general terms such as “disease”, “syndrome”, “deficit”, “complications”, “tumor”,
            unless they are part of a specific phrase ("breast cancer", "brain tumor", "metastatic breast cancer").
            4. Do not annotate biological processes such as “carcinogenesis” or “tumorigenesis”.
            5. Do not annotate organisms such as “human”, “bacterial”, unless in context they directly refer to a disease (“Epstein-Barr virus”).
            """,
    'annotation_reduced': """
        You are a careful medical expert identifying biomedical entities in text.
        You consider context carefully and output a list of entities in the exact order and wording as they appear:
        Format: ["entity_text">, "entity_text", ...].
        Annotate only valid, contextually correct entities."""}
# tool calling llama.cpp
PROMPT = {
    'genre_syn_generation_new' : """
        You are a {language} medical expert. For the given {condition}, produce a plausible medical context within a {genre}. 
        Use semantic memory retrieval, contextual inference, and perspective-taking to avoid generic or templated phrasing. 
        Generate exactly one sentence using:
        {condition} as a {place} sentence within the medical context of {genre}.        
        Avoid diagnostic boilerplate (e.g., “diagnosed with”). 
        Do not use the words {avoid_words}. Output plain {language} text only. \n\n""",    

    'kshot_entity': """
        You are a {language} medical expert. For the given {condition}, produce a plausible medical context within a {genre}. 
        Use semantic memory retrieval, contextual inference, and perspective-taking to avoid generic or templated phrasing. 
        Generate exactly one sentence using:
        {condition} as a {place} sentence within the medical context of {genre}.      
        Avoid diagnostic boilerplate (e.g., “diagnosed with”). 
        Do not use the words {avoid_words}. Output plain {language} text only.
        Use the following examples as inspiration for linguistic style and structure from provided sentence, part of speech tags (POS) and dependency parsing tags (DEP).
        Examples:
        {text}
        Your task is to produce just the sentence in {language}.
        Sentence:
        """,

    'kshot_no_entity': """
        You are a {language} medical expert generating realistic clinical language. 
        Infer a plausible clinical context within a {genre}. Use semantic memory retrieval, contextual inference, 
        and perspective-taking to avoid generic or templated phrasing. 
        Generate exactly one sentence as a {place} sentence within the medical context of {genre}.   
        Do not use the words {avoid_words}. Output plain {language} text only.
        Use the following examples as inspiration for linguistic style and structure from provided sentence, part of speech tags (POS) and dependency parsing tags (DEP).
        Examples:
        {text}
        Your task is to produce just the sentence in {language} that does not contain any disease or diagnosis.
        Sentence:
        """,
    'disease_annotation_reduced': """Find and extract diagnoses and diseases mentioned in the following text.
        Annotate only diagnoses and diseases — not symptoms, signs, tests, or treatments.
        A diagnosis identifies a specific disease or condition based on medical evaluation.
        A disease is a harmful deviation from normal function, caused by factors such as pathogens, genetics, or environment.
        Exclude general or vague terms (“disease”, “syndrome”, “tumor”) unless they are part of a specific phrase (e.g., “breast cancer”).
        Do not annotate biological processes (“carcinogenesis”) or organisms (“bacterial”) unless they directly refer to a disease 
        (e.g., “Epstein-Barr virus”). Use exact spans from the sentence, preserving their form and order. 
        use semantic memory retrieval, and contextual inference, 
        and analytical reasoning, and information processing and scientific reasoning to find entities.
        Output a list of strings: ["entity_text", "entity_text", ...]. If no valid entities are found, output an empty list: []. 
        A mention of {condition} in following text is possible.
        Find and extract diagnoses and diseases in format ["entity_text", "entity_text", ...]:
        Sentence: 
        {text}
        Entities:""",
    'disease_annotation_reduced_kshot': """Find and extract diagnoses and diseases mentioned in the following text.
        Annotate only diagnoses and diseases — not symptoms, signs, tests, or treatments.
        A diagnosis identifies a specific disease or condition based on medical evaluation.
        A disease is a harmful deviation from normal function, caused by factors such as pathogens, genetics, or environment.
        Exclude general or vague terms (“disease”, “syndrome”, “tumor”) unless they are part of a specific phrase (e.g., “breast cancer”).
        Do not annotate biological processes (“carcinogenesis”) or organisms (“bacterial”) unless they directly refer to a disease 
        (e.g., “Epstein-Barr virus”). Use exact spans from the sentence, preserving their form and order. 
        use semantic memory retrieval, and contextual inference, 
        and analytical reasoning, and information processing and scientific reasoning to find entities.
        Output a list of strings: ["entity_text", "entity_text", ...]. If no valid entities are found, output an empty list: []. 
        ##Examples:
        {kshot_text}
        ##Task:
        A mention of {condition} in following text is possible.
        Find and extract diagnoses and diseases in format ["entity_text", "entity_text", ...]:
        Sentence: 
        {text}
        Entities:"""
        }


class Role(Enum):
    NURSE = "nurse"
    DOCTOR = "doctor"
    MEDICAL_RESEARCHER = "medical researcher"
    PHYSICIAN = "physician"
    CLINICIAN = "clinician"
    MD = "medical doctor"
    SPECIALIST = "medical specialist"
    GENERAL_PRACTITIONER = "general practitioner"
    HEALTHCARE_PROFESSIONAL = "healthcare professional"
    MEDICAL_EXPERT = "medical expert"
    MD_SHORT = "MD"


class SentType(Enum):
    SIMPLE = "simple"
    COMPLEX = "complex"
    COMPOUND = "compound"
    COMPOUND_COMPLEX = "compound complex"
    

class Place(Enum):
    FIRST = "first"
    SECOND = "second"
    THIRD = "third"
    FOURTH = "fourth"
    FIFTH = "fifth"


class Genre(Enum):
    ABSTRACT = "medical paper abstract"
    ANAMNESIS_MD = "anamnesis from one MD to another MD"
    ANAMNESIS_PT = "anamnesis from MD to patient"
    LETTER = "medical letter"
    RESEARCH = "research paper"
    CASESTUDY = "patient case-study"
    NOTE = "medical note"
    REPORT = "medical report"
    SUMMARY = "medical summary"
    DISCHARGE = "discharge summary"
    PRESCRIPTION = "medical prescription"
    GUIDELINE = "medical guideline"
    REVIEW = "medical review article"
    DRUG_INSERTS = "drug inserts"
    PRODUCT_CHARACTERISTICS = "healthcare product characteristics"
    MEDICATION_LEAFLETS = "medication leaflets"
    RESEARCH_TITLE = "research paper title"
    CASESTUDY_TITLE = "patient case-study title"
    NOTE_TITLE = "medical note title"
    REPORT_TITLE = "medical report title"
    LETTER_TITLE = "medical letter title"



GENRE_GROUPS = {
    "general": [
        Genre.ABSTRACT,
        Genre.RESEARCH,
        Genre.REPORT,
        Genre.SUMMARY,
        Genre.REVIEW,
    ],
    "clinical": [
        Genre.ANAMNESIS_MD,
        Genre.ANAMNESIS_PT,
        Genre.LETTER,
        Genre.CASESTUDY,
        Genre.NOTE,
        Genre.DISCHARGE,
    ],
    "titles and leflets": [
        Genre.RESEARCH_TITLE,        # reuse allowed ✅
        Genre.CASESTUDY_TITLE,
        Genre.NOTE_TITLE,
        Genre.REPORT_TITLE,
        Genre.DRUG_INSERTS,
        Genre.PRODUCT_CHARACTERISTICS,
        Genre.MEDICATION_LEAFLETS,
       Genre.LETTER_TITLE,
    ],
    "guides": [
        Genre.REPORT,        # reuse allowed ✅
        Genre.PRESCRIPTION,
        Genre.GUIDELINE,
        Genre.REVIEW,
        Genre.CASESTUDY,
    ]
}

    
@dataclass
class PromptTemplate:
    """Template for system prompts with randomizable elements"""
    base_template: str
    role_variations: List[str] = field(default_factory=list)
    sent_type_variations: List[str] = field(default_factory=list)
    
    def format(self, role: str = None, sent_type: str = None, **kwargs) -> str:
        """Format the template with given or random values"""
        if role is None and self.role_variations:
            role = random.choice(self.role_variations)
        if sent_type is None and self.sent_type_variations:
            sent_type = random.choice(self.sent_type_variations)
        return self.base_template.format(role=role, sent_type=sent_type, **kwargs)

class PromptBuilder:
    """Advanced prompt builder with randomization and templating capabilities
    Args:
        sytem_templates: str key to dictionary: 'initial_prompt', 'role_prompt', 
            'role_sent_type_prompt', 'annotation'
        user_templates: str key to dictionary 'initial_prompt', 'genre_prompt',
             'disease_annotation_reduced', 'kshot_num_sent_entity'
            'kshot_no_entity', 'kshot_entity', 'genre_syn_generation_new'
    Output:
        
    """
    
    def __init__(self):
        self.system_templates = {
            # 'initial_prompt': PromptTemplate(
            #     base_template=SYSTEM_PROMPTS['basic'],
            # ),
            # 'role_prompt': PromptTemplate(
            #     base_template=SYSTEM_PROMPTS['role'],
            #     role_variations=[role.value for role in Role],
            # ),
            'role_sent_type_prompt': PromptTemplate(
                base_template=SYSTEM_PROMPTS['role_sent_type'],
                role_variations=[role.value for role in Role],
                sent_type_variations=[sent_type.value for sent_type in SentType],
            ),

            'annotation': PromptTemplate(
                base_template=SYSTEM_PROMPTS['annotation']
            ),
        }
        
        self.user_templates = {
            # 'initial_prompt': PROMPT['syn_generation'],
            # 'genre_prompt': PROMPT['genre_syn_generation'],
            # 'disease_annotation': PROMPT['disease_annotation'],
            'disease_annotation_reduced': PROMPT['disease_annotation_reduced'],
            'disease_annotation_reduced_kshot': PROMPT['disease_annotation_reduced_kshot'],

            # 'kshot_num_sent_entity':PROMPT['kshot_num_sent_entity'],
            'kshot_no_entity':PROMPT['kshot_no_entity'],
            'kshot_entity':PROMPT['kshot_entity'],
            'genre_syn_generation_new':PROMPT['genre_syn_generation_new'],
        }
        
        self.randomization_options = {
            'genre': [genre.value for genre in Genre],
            'sent_type':[sent_type.value for sent_type in SentType],
            'place': [place.value for place in Place],
            'first_sentence': [
                "The patient was diagnosed with",
                "Clinical evaluation revealed",
                "The diagnosis confirmed",
                "Medical assessment indicated",
                "The patient presents with",
                "Initial findings suggest",
                "The examination showed",
                "Diagnostic tests confirmed",
                "The clinical picture is consistent with",
                "The patient exhibits symptoms of",
                # "{}-year-old patient diagnosed with".format(random.randint(1, 100)),
                "{}-year-old patient diagnosed with".format(random.randint(35, 80)),
                "{}-year-old patient diagnosed with".format(random.randint(25, 90)),]
        }
    
    def get_system_prompt(self, template_key: str, role: str = None, sent_type: str = None,
                         randomize: bool = False, language: str = 'english',**kwargs) -> str:
        """Generate system prompt with optional randomization"""
        if template_key not in self.system_templates:
            raise ValueError(f"Template '{template_key}' not found. Available: {list(self.system_templates.keys())}")
        
        template = self.system_templates[template_key]
        if not randomize:
            # Use provided values or defaults
            role = role or (template.role_variations[0] if template.role_variations else "medical professional")
            sent_type = sent_type or (template.sent_type_variations[0] if template.sent_type_variations else "simple")
        return template.format(role=role, sent_type=sent_type, language=language, **kwargs)
    
    def get_user_prompt(self, template_key: str, condition: str, 
                        text: Optional[str] = None, randomize: bool = True, 
                        number_of_sentences: int = 1, language='english', avoid_words='', 
                        **kwargs) -> str:
        """Generate user prompt with optional randomization"""
        if template_key not in self.user_templates:
            raise ValueError(f"Template '{template_key}' not found. Available: {list(self.user_templates.keys())}")
        
        template = self.user_templates[template_key]
        # Add random elements if requested
        if randomize:
            if 'genre' in template and 'genre' not in kwargs:
                genre_group = kwargs.get("genre_group", "general")   
                kwargs['genre'] = select_genre(genre_group)
                kwargs['sent_type'] = random.choice(self.randomization_options['sent_type'])
                kwargs['place'] = random.choice(self.randomization_options['place'])
                kwargs['first_sentence'] = random.choice(self.randomization_options['first_sentence'])
        return template.format(condition=condition, text=text, 
                               number_of_sentences=number_of_sentences, 
                               language=language, avoid_words=avoid_words, **kwargs)

    def build_messages(self, system_template: str, 
                       user_template: str, 
                       condition: str, 
                       text: Optional[str] = "",
                      system_randomize: bool = True, 
                      user_randomize: bool = True,
                      system_kwargs: Dict = {}, 
                      user_kwargs: Dict = {},
                      number_of_sentences: int = 1,
                      language: str = 'english',
                      avoid_words: str = '',
                      genre_group: str = 'general',
                      kshot_text: str = '',
                      ) -> List[Dict[str, Any]]:
        """Build complete message array for API request"""
        
        system_content = self.get_system_prompt(
            system_template, randomize=system_randomize, language=language, **system_kwargs
        )
        user_content = self.get_user_prompt(
            user_template, condition, text=text, randomize=user_randomize, 
            number_of_sentences=number_of_sentences, language=language, 
            avoid_words=avoid_words, genre_group=genre_group, kshot_text=kshot_text,
              **user_kwargs
        )        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def add_custom_template(self, template_type: str, key: str, template: PromptTemplate):
        """Add custom template to the builder"""
        if template_type == 'system':
            self.system_templates[key] = template
        elif template_type == 'user':
            self.user_templates[key] = template
        else:
            raise ValueError("template_type must be 'system' or 'user'")



def select_genre(group: str) -> str:
    genres = GENRE_GROUPS.get(group, GENRE_GROUPS["general"])
    return random.choice(genres).value


def format_chat(messages: List[Dict[str, Any]]) -> str:
    """Format messages using the template compatible with llama.cpp server."""
    formatted = ""
    for i, msg in enumerate(messages):
        content = f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content'].strip()}<|eot_id|>"
        if i == 0:
            content = "<|begin_of_text|>" + content
        formatted += content
    # Add generation prompt for assistant
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return formatted


def message_request(args: argparse.Namespace, 
                    batch: Dict,
                    terms: List[str] = [],
                    prompt_builder: PromptBuilder = PromptBuilder(),
                    system_template: str = '',
                    user_template: str = '', 
                    max_tokens:int = None):
    """
    Improved message request function using PromptBuilder
    Args:
        args (argparse.Namespace): Command line arguments
        condition (str): The condition or term to include in the prompt
        prompt_builder (PromptBuilder): Instance of PromptBuilder to generate prompts
        system_template (str): Key for the system prompt template initial_prompt or role_prompt
        user_template (str): Key for the user prompt template initial_prompt or genre_prompt
    """

    messages = [prompt_builder.build_messages(
        system_template=system_template if system_template else item['system_template'],
        user_template=user_template if user_template else item['user_template'],
        condition=terms[i] if terms else item['term'][0],
        system_randomize=args.randomize_prompts if hasattr(args, 'randomize_prompts') else True,
        text=item['text_block'],
        user_randomize=args.randomize_prompts if hasattr(args, 'randomize_prompts') else True,
        number_of_sentences=getattr(args, 'num_sentences', 1), 
        language=getattr(args, 'language', 'english'),
        avoid_words=getattr(args, 'avoid_words', 'patient, individual, or subject'),
        genre_group=getattr(args, 'genre_group', 'general'),
        kshot_text=item.get('kshot_text', '')
    ) for i, item in enumerate(batch)]
    for i, message in enumerate(messages):
        args.logger.info(f'Generated {i} SYSTEM prompt:\n {message[0]["content"]}')
        args.logger.info(f'Generated {i} USER prompt:\n {message[1]["content"]}')
    prompt = [format_chat(message) for message in messages]
    # Send to llama.cpp HTTP server
    response = requests.post(
        f"{args.server_url}/completion",
        json={
            "prompt": prompt,
            "max_tokens": max_tokens if max_tokens else args.max_tokens,
            "temperature": args.temperature,
            "stop": ["<|eot_id|>"]
        })
    
    if args.verbose:
        args.logger.info('API response:')
        args.logger.info(f'Response status code: {response.status_code}')
        # args.logger.info(f'System prompt used: {messages[0]["content"][:100]}...')
        # args.logger.info(response.json()['content'])
    
    if response.status_code != 200:
        args.logger.info(f"Error: {response.status_code} - {response.text}")
        raise Exception(f"Request failed with status code {response.status_code}")
    
    return response


# Usage examples
if __name__ == "__main__":
    # Initialize prompt builder
    builder = PromptBuilder()
    print("\n" + "="*50 + "\n")
    print("Complete message structure:")
    for msg in messages:
        print(f"{msg['role'].upper()}:")
        print(msg['content'])
        print()
    
    condition = "flue"
    messages = builder.build_messages(
        system_template='annotation',
        user_template='disease_annotation_reduced',
        condition=condition,
        text="The patient was diagnosed with diabetes mellitus type 2 and hypertension."
    )
    
    print("Complete message structure:")
    for msg in messages:
        print(f"{msg['role'].upper()}:")
        print(msg['content'])
        print()
    
    print('+'*120)
    messages = builder.build_messages(
        system_template='role_sent_type_prompt',
        user_template='genre_syn_generation_new',
        condition=condition,
        text="The patient was diagnosed with diabetes mellitus type 2 and hypertension."
    )
    
    print("Complete message structure:")
    for msg in messages:
        print(f"{msg['role'].upper()}:")
        print(msg['content'])
        print()