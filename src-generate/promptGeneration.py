import random
import argparse
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

#TODO: clean up old prompts
SYSTEM_PROMPTS = {
    'basic':"""
        You are a carefull medical expert in writing you consider background knowledge 
        and medical experties when dealing with tasks you solve. You think about
        the task and how to output and produce real example of text simple sentence without decoration. 
        """,
    'role':"""
        You are a carefull {role} in writing you consider background knowledge 
        and medical experties when dealing with tasks you solve. You think about
        the task and how to output and produce real example of text simple sentence without decoration. 
        """,
    'annotation':"""You are a carefull medical expert in finding
        medical entities in text. You take context in to account when 
        searching for entities and output the desired structure a 
        list [<entity_text>, <entity_text>, ...]. 
        <entity_text> part of the sentence is the exact copy from the given sentence to analyse. If you find <entity_text> span that has a word 
        between the main entity point you transcribe the <entity_text> as read in sentence you are analysing. 
        You make shure you annotate only the correct entites. You need to carefully consider the entity order in sentence.
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
        Format: [<entity_text>, <entity_text>, ...].
        Annotate only valid, contextually correct entities."""}
# tool calling llama.cpp
PROMPT = {
    'syn_generation': """
        A diagnosis is the process of identifying a specific disease or condition 
        that is dependent of patient's signs, symptoms, and medical history. 
        A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
        structural or functional state of an organism, associated with specific signs and symptoms and 
        distinct from physical injury. It can be caused by various factors, including pathogens, 
        genetic or autoimmune dysfunction, or environmental influences.
        Diseases are generally characterized by a pathological process and 
        a specific set of symptoms, diseases can be grouped in Abnormal Condition, 
        Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
        Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, 
        Autoimmune Diseases, Degenerative Diseases.
        For given <disease> think about context in what kind of sentence could it occure
        your task is to generate just one a sentence containing the given <disease>, 
        you don't decorate or generate anything else. 
        You create a single sentence of text with the term used as it would be used in 
        medical document. Remember keep it simple. 
        DO not use word patient or individual or subject in your generated text. 
        Consider somethig you would see in an anamnesis or abstract or case-study or some other 
        medical document. Consider the sentence output that comes after sentence: 
        The patient diagnosed with {condition}.        
        Based on your medical expertise. 
        Your task is to generate next sentence containing following disease: {condition}. \n\n""",

    'genre_syn_generation': """
        A diagnosis is the process of identifying a specific disease or condition 
        that is dependent of patient's signs, symptoms, and medical history. 
        A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
        structural or functional state of an organism, associated with specific signs and symptoms and 
        distinct from physical injury. It can be caused by various factors, including pathogens, 
        genetic or autoimmune dysfunction, or environmental influences.
        Diseases are generally characterized by a pathological process and 
        a specific set of symptoms, diseases can be grouped in Abnormal Condition, 
        Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
        Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, 
        Autoimmune Diseases, Degenerative Diseases.
        For given <disease or diagnosis> think about context in what kind of sentence could it occure
        your task is to generate just one a sentence containing the given {condition}, 
        you don't decorate or generate anything else. 
        You create a single sentence of text with the entity {condition} used as it would be used in 
        medical document. Remember keep it simple. Do not use word patient or individual or subject in your generated text. 
        Consider somethig you would see in an {genre}. Consider the sentence output that comes after sentence: 
        The patient diagnosed with {condition}. 
        Based on your medical expertise. 
        Your task is to generate next sentence containing following disease or diagnosis <{condition}>. \n\n""",
        # genre = anamnesis, abstract, case-study
        # We want to check if there are additional diagnoses in produced sentence 
        # ('this is the part where there could be noise)
        # json output
        
    'kshot_genre_generation_with_entity': """
        Generate one simple sentence that naturally includes the disease or diagnosis {condition}, as it would appear in a {genre}.
        Requirements:
        Use {condition} exactly as written within the sentence.
        Keep the sentence factual, and contextually realistic for clinical or biomedical text.
        Output only the sentence — no explanations or extra text.
        Examples:
        {text}
        Your task is to produce just the sentence.
        Sentence:
        """,
    'kshot_genre_no_entity': """
        Generate one simple sentence that does NOT contain the disease or diagnosis, but resembles the syntax and tone of the examples.
        Requirements:
        Keep the sentence factual, and contextually realistic for clinical or biomedical text.
        Output only the sentence — no explanations or extra text. 
        Examples:
        {text}
        Your task is to produce just the sentence.
        Sentence:
        """,
        
    'kshot_num_sent_genre_entity': """Generate {number_of_sentences} sentences that contain the disease or diagnosis {condition}, as it would appear in a {genre}.
        Requirements:
        This is important, the output for "Entities: [{condition}]", you use <{condition}> exactly as written within one of the sentences.
        Keep the sentences factual, and contextually realistic for clinical or biomedical text.
        Output only sentences — no explanations or extra text or decorations. 
        Examples:
        {text}
        Your task is to produce just the sentences and the Entities: [] part, given the upper requirements.
        Sentences:
        """,
        
        
    'disease_annotation': """Find and extract diagnoses and diseases from the text and. 
        Your output should be a list of entities that is machine-readable using eval(): 
        [<entity_text>, <entity_text>, ...]. 
        Think about context of the text. As a medical expert, you know that patient symptoms, 
        expert-defined signs, diagnostic and medical tests, and treatments are NOT diagnoses or diseases. 
        You search for diagnoses and disease: 
        A diagnosis is the process of identifying a specific disease or condition 
        that is dependent of patient's signs, symptoms, and medical history. 
        A disease is a specific illness or disorder that affects the body. A disease is a harmful deviation from the normal 
        structural or functional state of an organism, associated with specific signs and symptoms and 
        distinct from physical injury. It can be caused by various factors, including pathogens, 
        genetic or autoimmune dysfunction, or environmental influences.
        Diseases are generally characterized by a pathological process and a specific set of symptoms, diseases can be 
        grouped in Abnormal Condition, Structural or Functional Impairment, Pathological Process, Not Immediate Injury, 
        Infectious Diseases, Non-Infectious Diseases, Hereditary (Genetic) Diseases, Autoimmune Diseases, Degenerative Diseases.
        Extract only diagnoses and dieases string sturctured [<entity_text>, <entity_text>, ...]
        if you don't find any entites output an empty list, i.e. []. Some form of {condition} is in the sentence.
        Diseases or diagnoses [<entity_text>,<entity_text>,...] must be the exact copy of spans from the sentence: 
        {text} \n\n          
         """,
         
    'disease_annotation_reduced': """Find and extract diagnoses and diseases mentioned in the following text.
        Output a Python-evaluable list of entity strings:
        Format: [<entity_text>, <entity_text>, ...].
        Guidelines:
        Annotate only diagnoses and diseases — not symptoms, signs, tests, or treatments.
        A diagnosis identifies a specific disease or condition based on medical evaluation.
        A disease is a harmful deviation from normal function, caused by factors such as pathogens, genetics, or environment.
        Exclude general or vague terms (“disease”, “syndrome”, “tumor”) unless they are part of a specific phrase (e.g., “breast cancer”).
        Do not annotate biological processes (“carcinogenesis”) or organisms (“bacterial”) unless they directly refer to a disease (e.g., “Epstein-Barr virus”).
        Use exact spans from the sentence, preserving their form and order.
        If no valid entities are found, output an empty list: [].
        Context: The sentence may mention a form of {condition}.
        Text to annotate:
        {text}"""
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


class Genre(Enum):
    ABSTRACT = "medical paper abstract"
    #ANAMNESIS_MD = "anamnesis from one MD to another MD"
    #ANAMNESIS_PT = "anamnesis from MD to patient"
    #LETTER = "medical letter"
    RESEARCH = "research paper"
    #CASESTUDY = "patient case-study"
    #NOTE = "medical note"
    REPORT = "medical report"
    SUMMARY = "medical summary"
    #DISCHARGE = "discharge summary"
    #PRESCRIPTION = "medical prescription"
    #GUIDELINE = "medical guideline"
    REVIEW = "medical review article"
    
    
@dataclass
class PromptTemplate:
    """Template for system prompts with randomizable elements"""
    base_template: str
    role_variations: List[str] = field(default_factory=list)
    
    def format(self, role: str = None, **kwargs) -> str:
        """Format the template with given or random values"""
        if role is None and self.role_variations:
            role = random.choice(self.role_variations)
        return self.base_template.format(role=role, **kwargs)

class PromptBuilder:
    """Advanced prompt builder with randomization and templating capabilities"""
    
    def __init__(self):
        self.system_templates = {
            'initial_prompt': PromptTemplate(
                base_template=SYSTEM_PROMPTS['basic'],
            ),
            'role_prompt': PromptTemplate(
                base_template=SYSTEM_PROMPTS['role'],
                role_variations=[role.value for role in Role],
            ),
            'annotation': PromptTemplate(
                base_template=SYSTEM_PROMPTS['annotation']
            ),
        }
        
        self.user_templates = {
            'initial_prompt': PROMPT['syn_generation'],
            'genre_prompt': PROMPT['genre_syn_generation'],
            'disease_annotation': PROMPT['disease_annotation'],
            'disease_annotation_reduced': PROMPT['disease_annotation_reduced'],
            'kshot_num_sent_genre_entity':PROMPT['kshot_num_sent_genre_entity'],
            'kshot_genre_no_entity':PROMPT['kshot_genre_no_entity'],
            'kshot_genre_generation_with_entity':PROMPT['kshot_genre_generation_with_entity'],
            
            
        }
        
        self.randomization_options = {
            'genre': [genre.value for genre in Genre],
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
                "{}-year-old patient diagnosed with".format(random.randint(1, 100)),
                "{}-year-old patient diagnosed with".format(random.randint(10, 100)),
                "{}-year-old patient diagnosed with".format(random.randint(25, 100)),]
        }
    
    def get_system_prompt(self, template_key: str, role: str = None, 
                         randomize: bool = False, **kwargs) -> str:
        """Generate system prompt with optional randomization"""
        if template_key not in self.system_templates:
            raise ValueError(f"Template '{template_key}' not found. Available: {list(self.system_templates.keys())}")
        
        template = self.system_templates[template_key]
        if not randomize:
            # Use provided values or defaults
            role = role or (template.role_variations[0] if template.role_variations else "medical professional")
        return template.format(role=role, **kwargs)
    
    def get_user_prompt(self, template_key: str, condition: str, 
                        text: Optional[str] = None, randomize: bool = True, 
                        number_of_sentences: int = 1, **kwargs) -> str:
        """Generate user prompt with optional randomization"""
        if template_key not in self.user_templates:
            raise ValueError(f"Template '{template_key}' not found. Available: {list(self.user_templates.keys())}")
        
        template = self.user_templates[template_key]
        
        # Add random elements if requested
        if randomize:
            if 'genre' in template and 'genre' not in kwargs:
                kwargs['genre'] = random.choice(self.randomization_options['genre'])

        return template.format(condition=condition, text=text, number_of_sentences=number_of_sentences, **kwargs)

    def build_messages(self, system_template: str, 
                       user_template: str, 
                       condition: str, 
                       text: Optional[str] = None,
                      system_randomize: bool = True, user_randomize: bool = True,
                      system_kwargs: Dict = None, user_kwargs: Dict = None,
                      number_of_sentences: int = 1,
                      ) -> List[Dict[str, Any]]:
        """Build complete message array for API request"""
        system_kwargs = system_kwargs or {}
        user_kwargs = user_kwargs or {}
        
        system_content = self.get_system_prompt(
            system_template, randomize=system_randomize, **system_kwargs
        )
        user_content = self.get_user_prompt(
            user_template, condition, text=text, randomize=user_randomize, 
            number_of_sentences=number_of_sentences, **user_kwargs
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
                    condition: str,
                   prompt_builder: PromptBuilder = PromptBuilder(), 
                   system_template: str = 'initial_prompt',
                   user_template: str = 'initial_prompt', 
                   text: Optional[str] = None) -> requests.Response:
    """
    Improved message request function using PromptBuilder
    Args:
        args (argparse.Namespace): Command line arguments
        condition (str): The condition or term to include in the prompt
        prompt_builder (PromptBuilder): Instance of PromptBuilder to generate prompts
        system_template (str): Key for the system prompt template initial_prompt or role_prompt
        user_template (str): Key for the user prompt template initial_prompt or genre_prompt
    """
    # Build messages using the prompt builder
    messages = prompt_builder.build_messages(
        system_template=system_template,
        user_template=user_template,
        condition=condition,
        system_randomize=args.randomize_prompts if hasattr(args, 'randomize_prompts') else True,
        text=text,
        user_randomize=args.randomize_prompts if hasattr(args, 'randomize_prompts') else True,
        number_of_sentences=getattr(args, 'num_sentences', 1), 
    )
    args.logger.info(f'Generated system prompt: {messages[0]["content"]}')
    args.logger.info(f'Generated user prompt: {messages[1]["content"]}')
    prompt = format_chat(messages)
    
    # Send to llama.cpp HTTP server
    response = requests.post(
        f"{args.server_url}/completion",
        json={
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["<|eot_id|>"]
        })
    
    if args.verbose:
        args.logger.info('Translation response:')
        args.logger.info(f'Response status code: {response.status_code}')
        args.logger.info(f'System prompt used: {messages[0]["content"][:100]}...')
        args.logger.info(response.json()['content'])
    
    if response.status_code != 200:
        args.logger.info(f"Error: {response.status_code} - {response.text}")
        raise Exception(f"Request failed with status code {response.status_code}")
    
    return response


# Usage examples
if __name__ == "__main__":
    # Initialize prompt builder
    builder = PromptBuilder()
    
    # Example 1: Random system prompt
    system_prompt = builder.get_system_prompt('initial_prompt')
    print("Random system prompt:")
    print(system_prompt)
    print("\n" + "="*50 + "\n")
    
    # Example 2: Specific role and task
    specific_prompt = builder.get_system_prompt(
        'role_prompt', 
        role='doctor', 
        randomize=False
    )
    print("Specific system prompt:")
    print(specific_prompt)
    print("\n" + "="*50 + "\n")
    
    # Example 3: Complete message building
    condition = "diabetes mellitus type 2"
    messages = builder.build_messages(
        system_template='role_prompt',
        user_template='initial_prompt',
        condition=condition
    )
    condition = "diabetes mellitus type 2"
    messages = builder.build_messages(
        system_template='role_prompt',
        user_template='kshot_genre_generation',
        condition=condition,
    )
    print("Complete message structure:")
    for msg in messages:
        print(f"{msg['role'].upper()}:")
        print(msg['content'])
        print()
    
    # Example 4: Adding custom template
    # custom_template = PromptTemplate(
    #     base_template="You are a {role} with {years} years of experience in {specialty}.",
    #     role_variations=["senior physician", "consultant", "specialist"],
    # )
    # builder.add_custom_template('system', 'custom_expert', custom_template)
    
    custom_prompt = builder.get_system_prompt(
        'initial_prompt', 
        role='clinician', 
        randomize=False
    )
    messages = builder.build_messages(
        system_template='role_prompt',
        user_template='genre_prompt',
        condition=condition
    )
    print("Custom template result:")
    print(custom_prompt)
    
    messages = builder.build_messages(
        system_template='annotation',
        user_template='disease_annotation',
        condition=condition,
        text="The patient was diagnosed with diabetes mellitus type 2 and hypertension."
    )
    
    print("Complete message structure:")
    for msg in messages:
        print(f"{msg['role'].upper()}:")
        print(msg['content'])
        print()