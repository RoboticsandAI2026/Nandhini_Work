# ============================== #
# ✅ Install Dependencies (only if not already installed)
# ============================== #



# ============================== #
# ✅ Imports
# ============================== #
import json
import torch
import joblib
from gpt_arc import GPTModel, download_and_load_gpt2, load_weights_into_gpt, generate, text_to_token_ids, token_ids_to_text
import tiktoken

# ============================== #
# ✅ Load Models (ML & LLM)
# ============================== #
ml_model = joblib.load("logistic_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_layers": 12,
    "n_heads": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
llm_model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(llm_model, params)

fine_tuned_path = "Fine_tuned_updrs_model.pth"
checkpoint = torch.load(fine_tuned_path, map_location="cpu")
llm_model.load_state_dict(checkpoint["Fine_tuned_model_state_dict"], strict=False)
llm_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm_model.to(device)
tokenizer = tiktoken.get_encoding("gpt2")

# ============================== #
# ✅ Load 59 UPDRS Questions
questions = [
    {"section": "Part I", "question_id": "1", "title": "SLEEP PROBLEMS", "prompt": "Over the past week, have you had trouble going to sleep at night or staying asleep through the night? Consider how rested you felt after waking up in the morning.", "choices": { "0": "Normal: No problems.", "1": "Slight: Sleep problems are present but usually do not cause trouble getting a full night of sleep.", "2": "Mild: Sleep problems usually cause some difficulties getting a full night of sleep.", "3": "Moderate: Sleep problems cause a lot of difficulties getting a full night of sleep, but I still usually sleep for more than half the night.", "4": "Severe: I usually do not sleep for most of the night."}},
    {"section": "Part I", "question_id": "2", "title": "DAYTIME SLEEPINESS", "prompt": "Over the past week, have you had trouble staying awake during the daytime?", "choices": { "0": "Normal: No daytime sleepiness.", "1": "Slight: Daytime sleepiness occurs, but I can resist and I stay awake.", "2": "Mild: Sometimes I fall asleep when alone and relaxing. For example, while reading or watching TV.", "3": "Moderate: I sometimes fall asleep when I should not. For example, while eating or talking with other people.", "4": "Severe: I often fall asleep when I should not. For example, while eating or talking with other people."}},
    {"section": "Part I", "question_id": "3", "title": "PAIN AND OTHER SENSATIONS", "prompt": "Over the past week, have you had uncomfortable feelings in your body like pain, aches, tingling, or cramps?", "choices": { "0": "Normal: No uncomfortable feelings.", "1": "Slight: I have these feelings. However, I can do things and be with other people without difficulty.", "2": "Mild: These feelings cause some problems when I do things or am with other people.", "3": "Moderate: These feelings cause a lot of problems, but they do not stop me from doing things or being with other people.", "4": "Severe: These feelings stop me from doing things or being with other people."}},
    {"section": "Part I", "question_id": "4", "title": "URINARY PROBLEMS", "prompt": "Over the past week, have you had trouble with urine control? For example, an urgent need to urinate, a need to urinate too often, or urine accidents?", "choices": { "0": "Normal: No urine control problems.", "1": "Slight: I need to urinate often or urgently. However, these problems do not cause difficulties with my daily activities.", "2": "Mild: Urine problems cause some difficulties with my daily activities. However, I do not have urine accidents.", "3": "Moderate: Urine problems cause a lot of difficulties with my daily activities, including urine accidents.", "4": "Severe: I cannot control my urine and use a protective garment or have a bladder tube."}},
    {"section": "Part I", "question_id": "5", "title": "CONSTIPATION PROBLEMS", "prompt": "Over the past week have you had constipation troubles that cause you difficulty moving your bowels?", "choices": { "0": "Normal: No constipation.", "1": "Slight: I have been constipated. I use extra effort to move my bowels. However, this problem does not disturb my activities or my being comfortable.", "2": "Mild: Constipation causes me to have some troubles doing things or being comfortable.", "3": "Moderate: Constipation causes me to have a lot of trouble doing things or being comfortable. However, it does not stop me from doing anything.", "4": "Severe: I usually need physical help from someone else to empty my bowels."}},
    {"section": "Part I", "question_id": "6", "title": "LIGHT HEADEDNESS ON STANDING", "prompt": "Over the past week, have you felt faint, dizzy, or foggy when you stand up after sitting or lying down?", "choices": { "0": "Normal: No dizzy or foggy feelings.", "1": "Slight: Dizzy or foggy feelings occur. However, they do not cause me troubles doing things.", "2": "Mild: Dizzy or foggy feelings cause me to hold on to something, but I do not need to sit or lie back down.", "3": "Moderate: Dizzy or foggy feelings cause me to sit or lie down to avoid fainting or falling.", "4": "Severe: Dizzy or foggy feelings cause me to fall or faint."}},
    {"section": "Part I", "question_id": "7", "title": "FATIGUE", "prompt": "Over the past week, have you usually felt fatigued? This feeling is not part of being sleepy or sad.", "choices": { "0": "Normal: No fatigue.", "1": "Slight: Fatigue occurs. However it does not cause me troubles doing things or being with people.", "2": "Mild: Fatigue causes me some troubles doing things or being with people.", "3": "Moderate: Fatigue causes me a lot of troubles doing things or being with people. However, it does not stop me from doing anything.", "4": "Severe: Fatigue stops me from doing things or being with people."}},

    {"section": "Part II", "question_id": "8", "title": "SPEECH", "prompt": "Over the past week, have you had problems with your speech?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: My speech is soft, slurred or uneven, but it does not cause others to ask me to repeat myself.", "2": "Mild: My speech causes people to ask me to occasionally repeat myself, but not every day.", "3": "Moderate: My speech is unclear enough that others ask me to repeat myself every day even though most of my speech is understood.", "4": "Severe: Most or all of my speech cannot be understood."}},
    {"section": "Part II", "question_id": "9", "title": "SALIVA AND DROOLING", "prompt": "Over the past week, have you usually had too much saliva during when you are awake or when you sleep?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I have too much saliva, but do not drool.", "2": "Mild: I have some drooling during sleep, but none when I am awake.", "3": "Moderate: I have some drooling when I am awake, but I usually do not need tissues or a handkerchief.", "4": "Severe: I have so much drooling that I regularly need to use tissues or a handkerchief to protect my clothes."}},
    {"section": "Part II", "question_id": "10", "title": "CHEWING AND SWALLOWING", "prompt": "Over the past week, have you usually had problems swallowing pills or eating meals? Do you need your pills cut or crushed or your meals to be made soft, chopped, or blended to avoid choking?", "choices": { "0": "Normal: No problems.", "1": "Slight: I am aware of slowness in my chewing or increased effort at swallowing, but I do not choke or need to have my food specially prepared.", "2": "Mild: I need to have my pills cut or my food specially prepared because of chewing or swallowing problems, but I have not choked over the past week.", "3": "Moderate: I choked at least once in the past week.", "4": "Severe: Because of chewing and swallowing problems, I need a feeding tube."}},
    {"section": "Part II", "question_id": "11", "title": "EATING TASKS", "prompt": "Over the past week, have you usually had troubles handling your food and using eating utensils? For example, do you have trouble handling finger foods or using forks, knives, spoons, chopsticks?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am slow, but I do not need any help handling my food and have not had food spills while eating.", "2": "Mild: I am slow with my eating and have occasional food spills. I may need help with a few tasks such as cutting meat.", "3": "Moderate: I need help with many eating tasks but can manage some alone.", "4": "Severe: I need help for most or all eating tasks."}},
    {"section": "Part II", "question_id": "12", "title": "DRESSING", "prompt": "Over the past week, have you usually had problems dressing? For example, are you slow or do you need help with buttoning, using zippers, putting on or taking off your clothes or jewelry?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am slow, but I do not need help.", "2": "Mild: I am slow and need help for a few dressing tasks (buttons, bracelets).", "3": "Moderate: I need help for many dressing tasks.", "4": "Severe: I need help for most or all dressing tasks."}},
    {"section": "Part II", "question_id": "13", "title": "HYGIENE", "prompt": "Over the past week, have you usually been slow or do you need help with washing, bathing, shaving, brushing teeth, combing your hair, or with other personal hygiene?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am slow, but I do not need any help.", "2": "Mild: I need someone else to help me with some hygiene tasks.", "3": "Moderate: I need help for many hygiene tasks.", "4": "Severe: I need help for most or all of my hygiene tasks."}},
    {"section": "Part II", "question_id": "14", "title": "HANDWRITING", "prompt": "Over the past week, have people usually had trouble reading your handwriting?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: My writing is slow, clumsy or uneven, but all words are clear.", "2": "Mild: Some words are unclear and difficult to read.", "3": "Moderate: Many words are unclear and difficult to read.", "4": "Severe: Most or all words cannot be read."}},
    {"section": "Part II", "question_id": "15", "title": "DOING HOBBIES AND OTHER ACTIVITIES", "prompt": "Over the past week, have you usually had trouble doing your hobbies or other things that you like to do?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am a bit slow but do these activities easily.", "2": "Mild: I have some difficulty doing these activities.", "3": "Moderate: I have major problems doing these activities, but still do most.", "4": "Severe: I am unable to do most or all of these activities."}},
    {"section": "Part II", "question_id": "16", "title": "TURNING IN BED", "prompt": "Over the past week, do you usually have trouble turning over in bed?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I have a bit of trouble turning, but I do not need any help.", "2": "Mild: I have a lot of trouble turning and need occasional help from someone else.", "3": "Moderate: To turn over I often need help from someone else.", "4": "Severe: I am unable to turn over without help from someone else."}},
    {"section": "Part II", "question_id": "17", "title": "TREMOR", "prompt": "Over the past week, have you usually had shaking or tremor?", "choices": { "0": "Normal: Not at all. I have no shaking or tremor.", "1": "Slight: Shaking or tremor occurs but does not cause problems with any activities.", "2": "Mild: Shaking or tremor causes problems with only a few activities.", "3": "Moderate: Shaking or tremor causes problems with many of my daily activities.", "4": "Severe: Shaking or tremor causes problems with most or all activities."}},
    {"section": "Part II", "question_id": "18", "title": "GETTING OUT OF BED, A CAR, OR A DEEP CHAIR", "prompt": "Over the past week, have you usually had trouble getting out of bed, a car seat, or a deep chair?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am slow or awkward, but I usually can do it on my first try.", "2": "Mild: I need more than one try to get up or need occasional help.", "3": "Moderate: I sometimes need help to get up, but most times I can still do it on my own.", "4": "Severe: I need help most or all of the time."}},
    {"section": "Part II", "question_id": "19", "title": "WALKING AND BALANCE", "prompt": "Over the past week, have you usually had problems with balance and walking?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I am slightly slow or may drag a leg. I never use a walking aid.", "2": "Mild: I occasionally use a walking aid, but I do not need any help from another person.", "3": "Moderate: I usually use a walking aid (cane, walker) to walk safely without falling. However, I do not usually need the support of another person.", "4": "Severe: I usually use the support of another person to walk safely without falling."}},
    {"section": "Part II", "question_id": "20", "title": "FREEZING", "prompt": "Over the past week, on your usual day when walking, do you suddenly stop or freeze as if your feet are stuck to the floor?", "choices": { "0": "Normal: Not at all (no problems).", "1": "Slight: I briefly freeze, but I can easily start walking again. I do not need help from someone else or a walking aid (cane or walker) because of freezing.", "2": "Mild: I freeze and have trouble starting to walk again, but I do not need someone’s help or a walking aid (cane or walker) because of freezing.", "3": "Moderate: When I freeze I have a lot of trouble starting to walk again and, because of freezing, I sometimes need to use a walking aid or need someone else’s help.", "4": "Severe: Because of freezing, most or all of the time, I need to use a walking aid or someone’s help."}},


    {"section": "Part III", "question_id": "21", "title": "SPEECH", "prompt": "Listen to the patient's free-flowing speech. Evaluate volume, modulation, and clarity.", "choices": { "0": "Normal: No speech problems.", "1": "Slight: Loss of modulation, diction, or volume, but all words easy to understand.", "2": "Mild: Loss of modulation, diction, or volume, with a few words unclear, but sentences easy to follow.", "3": "Moderate: Speech is difficult to understand to the point that some, but not most, sentences are poorly understood.", "4": "Severe: Most speech is difficult to understand or unintelligible."}},
    {"section": "Part III", "question_id": "22", "title": "FACIAL EXPRESSION", "prompt": "Observe the patient for masked facies or loss of facial expression.", "choices": { "0": "Normal facial expression.", "1": "Slight: Minimal masked facies, only decreased frequency of blinking.", "2": "Mild: Masked facies present in lower face, fewer mouth movements.", "3": "Moderate: Masked facies with lips parted some of the time at rest.", "4": "Severe: Masked facies with lips parted most of the time at rest."}},
    {"section": "Part III", "question_id": "23", "title": "RIGIDITY - NECK", "prompt": "Judge rigidity by passive movement of the neck.", "choices": { "0": "Normal: No rigidity.", "1": "Slight: Rigidity only detected with activation maneuver.", "2": "Mild: Rigidity without activation maneuver; full range of motion easily achieved.", "3": "Moderate: Rigidity without activation maneuver; full range of motion achieved with effort.", "4": "Severe: Rigidity without activation maneuver and full range of motion not achieved."}},
    {"section": "Part III", "question_id": "24", "title": "RIGIDITY - RIGHT UPPER LIMB", "prompt": "Judge rigidity by passive movement of the right upper limb.", "choices": { "0": "Normal: No rigidity.", "1": "Slight: Rigidity only detected with activation maneuver.", "2": "Mild: Rigidity without activation maneuver; full range of motion easily achieved.", "3": "Moderate: Rigidity without activation maneuver; full range of motion achieved with effort.", "4": "Severe: Rigidity without activation maneuver and full range of motion not achieved."}},
    {"section": "Part III", "question_id": "25", "title": "RIGIDITY - LEFT UPPER LIMB", "prompt": "Judge rigidity by passive movement of the left upper limb.", "choices": { "0": "Normal: No rigidity.", "1": "Slight: Rigidity only detected with activation maneuver.", "2": "Mild: Rigidity without activation maneuver; full range of motion easily achieved.", "3": "Moderate: Rigidity without activation maneuver; full range of motion achieved with effort.", "4": "Severe: Rigidity without activation maneuver and full range of motion not achieved."}},
    {"section": "Part III", "question_id": "26", "title": "RIGIDITY - RIGHT LOWER LIMB", "prompt": "Judge rigidity by passive movement of the right lower limb.", "choices": { "0": "Normal: No rigidity.", "1": "Slight: Rigidity only detected with activation maneuver.", "2": "Mild: Rigidity without activation maneuver; full range of motion easily achieved.", "3": "Moderate: Rigidity without activation maneuver; full range of motion achieved with effort.", "4": "Severe: Rigidity without activation maneuver and full range of motion not achieved."}},
    {"section": "Part III", "question_id": "27", "title": "RIGIDITY - LEFT LOWER LIMB", "prompt": "Judge rigidity by passive movement of the left lower limb.", "choices": { "0": "Normal: No rigidity.", "1": "Slight: Rigidity only detected with activation maneuver.", "2": "Mild: Rigidity without activation maneuver; full range of motion easily achieved.", "3": "Moderate: Rigidity without activation maneuver; full range of motion achieved with effort.", "4": "Severe: Rigidity without activation maneuver and full range of motion not achieved."}},
    {"section": "Part III", "question_id": "28", "title": "FINGER TAPPING - RIGHT", "prompt": "Right hand: Instruct patient to tap the index finger on the thumb 10 times as quickly and as big as possible.", "choices": { "0": "Normal: No problems.", "1": "Slight: 1–2 interruptions/hesitations, slight slowing, amplitude decrements near end.", "2": "Mild: 3–5 interruptions, mild slowing, amplitude decrements midway.", "3": "Moderate: >5 interruptions or 1 freeze, moderate slowing, amplitude loss early.", "4": "Severe: Cannot or can barely perform the task."}},
    {"section": "Part III", "question_id": "29", "title": "FINGER TAPPING - LEFT", "prompt": "Left hand: Instruct patient to tap the index finger on the thumb 10 times as quickly and as big as possible.", "choices": { "0": "Normal: No problems.", "1": "Slight: 1–2 interruptions/hesitations, slight slowing, amplitude decrements near end.", "2": "Mild: 3–5 interruptions, mild slowing, amplitude decrements midway.", "3": "Moderate: >5 interruptions or 1 freeze, moderate slowing, amplitude loss early.", "4": "Severe: Cannot or can barely perform the task."}},
    {"section": "Part III", "question_id": "30", "title": "HAND MOVEMENTS - RIGHT", "prompt": "Right hand: Open and close your hand as fully and as quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude; may miss one closure.", "3": "Moderate: Marked slowing; marked reduction in amplitude; misses several closures.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "31", "title": "HAND MOVEMENTS - LEFT", "prompt": "Left hand: Open and close your hand as fully and as quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude; may miss one closure.", "3": "Moderate: Marked slowing; marked reduction in amplitude; misses several closures.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "32", "title": "PRONATION-SUPINATION - RIGHT", "prompt": "Right hand: Tap the palm and back of your hand on your lap as fully and quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Slight slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude.", "3": "Moderate: Marked slowing or amplitude loss; misses several rotations.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "33", "title": "PRONATION-SUPINATION - LEFT", "prompt": "Left hand: Tap the palm and back of your hand on your lap as fully and quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Slight slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude.", "3": "Moderate: Marked slowing or amplitude loss; misses several rotations.", "4": "Severe: Can barely perform or unable."}},

    {"section": "Part III", "question_id": "34", "title": "TOE TAPPING - RIGHT", "prompt": "Right foot: Tap your big toe on the ground as big and as fast as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate amplitude loss.", "3": "Moderate: Marked slowing or loss; misses several taps.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "35", "title": "TOE TAPPING - LEFT", "prompt": "Left foot: Tap your big toe on the ground as big and as fast as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate amplitude loss.", "3": "Moderate: Marked slowing or loss; misses several taps.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "36", "title": "LEG AGILITY - RIGHT", "prompt": "Right leg: Raise and lower your leg as fully and quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude.", "3": "Moderate: Marked slowing; misses several raises.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "37", "title": "LEG AGILITY - LEFT", "prompt": "Left leg: Raise and lower your leg as fully and quickly as possible for 10 seconds.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor slowing or reduction in amplitude.", "2": "Mild: Mild slowing; moderate reduction in amplitude.", "3": "Moderate: Marked slowing; misses several raises.", "4": "Severe: Can barely perform or unable."}},
    {"section": "Part III", "question_id": "38", "title": "ARISING FROM CHAIR", "prompt": "Ask the patient to stand up from a straight-backed chair with arms folded.", "choices": { "0": "Normal: Able to rise quickly without help.", "1": "Slight: Slow or may use hands once.", "2": "Mild: Pushes with hands more than once.", "3": "Moderate: May need several attempts or minor assistance.", "4": "Severe: Unable to rise without major help."}},
    {"section": "Part III", "question_id": "39", "title": "GAIT", "prompt": "Have the patient walk, observe stride, arm swing, stability.", "choices": { "0": "Normal: No problems.", "1": "Slight: Minor changes but does not affect walking.", "2": "Mild: Mildly impaired; short steps but does not need aid.", "3": "Moderate: Needs a walking aid but not a person.", "4": "Severe: Needs assistance of another person to walk or cannot walk."}},
    {"section": "Part III", "question_id": "40", "title": "FREEZING OF GAIT", "prompt": "Observe the patient for freezing episodes while walking.", "choices": { "0": "Normal: No freezing.", "1": "Slight: Rare freezing; does not affect gait.", "2": "Mild: Occasional freezing; mild gait disturbance.", "3": "Moderate: Frequent freezing; needs some help.", "4": "Severe: Freezes frequently and requires substantial help."}},
    {"section": "Part III", "question_id": "41", "title": "POSTURAL STABILITY", "prompt": "Test patient's ability to recover from a gentle pull backward on the shoulders.", "choices": { "0": "Normal: Recovers unaided with 1–2 steps.", "1": "Slight: 3–5 steps but recovers unaided.", "2": "Mild: >5 steps, but does not fall.", "3": "Moderate: Would fall if not caught.", "4": "Severe: Very unstable, spontaneous loss of balance."}},
    {"section": "Part III", "question_id": "42", "title": "POSTURE", "prompt": "Observe the patient's posture while standing and walking.", "choices": { "0": "Normal: Erect.", "1": "Slight: Slightly stooped, can correct easily.", "2": "Mild: Moderately stooped, can correct with effort.", "3": "Moderate: Markedly stooped, cannot correct.", "4": "Severe: Very marked flexion or unstable posture."}},
    {"section": "Part III", "question_id": "43", "title": "GLOBAL SPONTANEITY OF MOVEMENT (BRADYKINESIA)", "prompt": "Assess the general slowness and decrease in amplitude of movement throughout the body.", "choices": { "0": "Normal: No slowness.", "1": "Slight: Slight, barely noticeable.", "2": "Mild: Obvious slowing.", "3": "Moderate: Severe slowness, impairs function.", "4": "Severe: Marked slowness, barely moves or cannot."}},




    {"section": "Part III", "question_id": "44", "title": "POSTURAL TREMOR OF RIGHT HAND", "prompt": "With arms extended, observe for postural tremor in the right hand.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "45", "title": "POSTURAL TREMOR OF LEFT HAND", "prompt": "With arms extended, observe for postural tremor in the left hand.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},

    {"section": "Part III", "question_id": "46", "title": "KINETIC TREMOR OF RIGHT HAND", "prompt": "Right hand: Ask the patient to touch their nose then the rater’s finger repeatedly. Observe for tremor.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "47", "title": "KINETIC TREMOR OF LEFT HAND", "prompt": "Left hand: Ask the patient to touch their nose then the rater’s finger repeatedly. Observe for tremor.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},

    {"section": "Part III", "question_id": "48", "title": "REST TREMOR - RIGHT UPPER EXTREMITY", "prompt": "With arms relaxed in lap, observe rest tremor in right upper limb.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "49", "title": "REST TREMOR - LEFT UPPER EXTREMITY", "prompt": "With arms relaxed in lap, observe rest tremor in left upper limb.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "50", "title": "REST TREMOR - RIGHT LOWER EXTREMITY", "prompt": "With legs relaxed, observe rest tremor in right lower limb.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "51", "title": "REST TREMOR - LEFT LOWER EXTREMITY", "prompt": "With legs relaxed, observe rest tremor in left lower limb.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},
    {"section": "Part III", "question_id": "52", "title": "REST TREMOR - LIP OR JAW", "prompt": "Observe the patient’s face at rest for tremor in the lips or jaw.", "choices": { "0": "Normal: No tremor.", "1": "Slight: <1 cm amplitude.", "2": "Mild: 1–3 cm amplitude.", "3": "Moderate: 3–10 cm amplitude.", "4": "Severe: >10 cm amplitude."}},

    {"section": "Part III", "question_id": "53", "title": "CONSTANCY OF REST TREMOR", "prompt": "Estimate the percentage of time rest tremor is present during observation (e.g., during different positions and activities).", "choices": { "0": "Normal: No rest tremor.", "1": "Slight: Present <25% of the time.", "2": "Mild: Present 26–50% of the time.", "3": "Moderate: Present 51–75% of the time.", "4": "Severe: Present >75% of the time."}},


    {"section": "Part IV", "question_id": "54", "title": "TIME SPENT WITH DYSKINESIAS", "prompt": "Over the past week, how much of your waking day did you have wiggling, twitching, or jerking movements (not tremor or cramps)?", "choices": { "0": "Normal: No dyskinesias.", "1": "Slight: ≤25% of waking day.", "2": "Mild: 26–50% of waking day.", "3": "Moderate: 51–75% of waking day.", "4": "Severe: >75% of waking day."}},
    {"section": "Part IV", "question_id": "55", "title": "FUNCTIONAL IMPACT OF DYSKINESIAS", "prompt": "Did the wiggling/jerking movements interfere with doing things or being with people?", "choices": { "0": "Normal: No impact.", "1": "Slight: Impact on a few activities, but you usually perform all.", "2": "Mild: Impact on many activities, but you usually perform all.", "3": "Moderate: Usually do not perform some activities or social interactions during episodes.", "4": "Severe: Usually do not perform most activities/social interactions during episodes."}},
    {"section": "Part IV", "question_id": "56", "title": "TIME SPENT IN THE OFF STATE", "prompt": "Over the past week, what percentage of your waking day was spent in 'OFF' state (low, slow, or bad time despite medication)?", "choices": { "0": "Normal: No OFF time.", "1": "Slight: ≤25% of waking day.", "2": "Mild: 26–50% of waking day.", "3": "Moderate: 51–75% of waking day.", "4": "Severe: >75% of waking day."}},
    {"section": "Part IV", "question_id": "57", "title": "FUNCTIONAL IMPACT OF FLUCTUATIONS", "prompt": "Did OFF periods interfere with your ability to do things or participate socially compared to ON periods?", "choices": { "0": "Normal: No impact.", "1": "Slight: Impact on a few activities, but you usually perform all.", "2": "Mild: Impact on many activities, but you usually perform all.", "3": "Moderate: Usually do not perform some activities or social interactions during OFF.", "4": "Severe: Usually do not perform most activities/social interactions during OFF."}},
    {"section": "Part IV", "question_id": "58", "title": "COMPLEXITY OF MOTOR FLUCTUATIONS", "prompt": "Are your OFF periods predictable (certain time, activity, or unpredictable)?", "choices": { "0": "Normal: No motor fluctuations.", "1": "Slight: OFF times are predictable all or almost all the time (>75%).", "2": "Mild: OFF times are predictable most of the time (51–75%).", "3": "Moderate: OFF times are predictable some of the time (26–50%).", "4": "Severe: OFF episodes are rarely predictable (≤25%)."}},
    {"section": "Part IV", "question_id": "59", "title": "PAINFUL OFF-STATE DYSTONIA", "prompt": "During OFF periods, what percentage of the time do you experience painful cramps or spasms (dystonia)?", "choices": { "0": "Normal: No dystonia or NO OFF TIME.", "1": "Slight: ≤25% of time in OFF state.", "2": "Mild: 26–50% of time in OFF state.", "3": "Moderate: 51–75% of time in OFF state.", "4": "Severe: >75% of time in OFF state."}},
]



# ============================== #
# ✅ Functions
# ============================== #
def ask_questionnaire():
    responses = {}
    print("\n🧠 Parkinson’s UPDRS Questionnaire")
    for q in questions:
        print(f"\nQ{q['question_id']}: {q['prompt']}")
        for k, v in q['choices'].items():
            print(f"{k}: {v}")
        while True:
            ans = input("Enter your response (0–4): ").strip()
            if ans in q['choices']:
                responses[q['question_id']] = int(ans)
                break
            else:
                print("❌ Invalid input. Please enter a number from 0–4.")
    return responses

def compute_updrs_score(responses):
    return sum(responses.values())

def predict_pd_status(updrs_score):
    input_score = [[updrs_score]]
    prediction = ml_model.predict(input_score)[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return label

def format_instruction(updrs_score):
    return f"UPDRS score: {updrs_score}"

def format_input(instruction):
    return (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}"
    )

def get_llm_response(instruction):
    input_text = format_input(instruction)
    encoded = text_to_token_ids(input_text, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=llm_model,
            idx=encoded,
            max_new_tokens=50,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
    output = token_ids_to_text(token_ids, tokenizer)
    return output.replace(input_text, "").replace("### Response:", "").strip()

# ============================== #
# ✅ Full Inference Pipeline
# ============================== #
print("\n🚀 Parkinson’s Diagnosis Assistant")

responses = ask_questionnaire()
updrs_score = compute_updrs_score(responses)
prediction = predict_pd_status(updrs_score)
llm_instruction = format_instruction(updrs_score)
llm_output = get_llm_response(llm_instruction)

print("\n==============================")
print(f"🔢 Total UPDRS Score: {updrs_score}")
print(f"🧾 ML Prediction: {prediction}")
print(f"🤖 LLM Recommendation: {llm_output}")
print("==============================")



