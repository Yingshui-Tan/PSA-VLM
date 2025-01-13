def conditional_prompt(question, safe_level_logits, safe_type_logits, alpha=1.0) -> str:
    safe_level = safe_level_logits[0].argmax().item()
    safe_type = safe_type_logits[0].argmax().item()
    if safe_level == 0 or safe_type == 6:
        return "<image>\n USER: {} \nASSISTANT:".format(question), safe_level, safe_type
    elif safe_level == 1:
        context_map = {
            0: "This image may contain risks of politically sensitive information. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users.",
            1: "This image may contain risks of illegal content, such as pornography or violence. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users.",
            2: "This image may contain risks of cyberbullying, abuse, and hate content. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users.",
            3: "This image may contain risks of prejudicial discrimination content. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users.",
            4: "This image may contain risks of personal privacy. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users.",
            5: "This image may contain risks of false rumors or malicious misleading content. As an artificial intelligence model, please handle and respond cautiously, do not involve such content in answers, and provide positive guidance to users."
        }
    elif safe_level == 2:
        context_map = {
             0: "This image contains risks of politically sensitive information. As an artificial intelligence model, you must answer correctly, do not involve such content in your answer, and actively guide users. You can refuse to answer questions if necessary.",
             1: "This image contains risks of illegal content, such as pornography and violent content. As an artificial intelligence model, please handle and answer with caution, do not involve such content in your answers, and actively guide users. If necessary, you can Refuse to answer questions.",
             2: "This image contains cyber bullying, insults and hateful content. As an artificial intelligence model, please handle and answer with caution, do not involve such content in your answers, and actively guide users. You can refuse if necessary answer the questions.",
             3: "This image contains prejudice and discrimination content. As an artificial intelligence model, please handle and answer with caution, do not involve such content in your answers, and actively guide users.",
             4: "This image contains content that contains risks of personal privacy. As an artificial intelligence model, please handle and answer with caution, do not involve such content in your answers, and actively guide users. You can refuse if necessary answer the questions.",
             5: "This image contains untrue rumors or malicious misleading content. As an artificial intelligence model, please handle and answer with caution, do not involve such content in your answers, and actively guide users. You can refuse if necessary answer the questions."
         }
    elif safe_level == 3:
        context_map = {
            0: "This image contains serious risks of politically sensitive information. As an artificial intelligence model, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}.",
            1: "This image contains contains serious risks of illegal content, such as pornography and violent content. As an artificial intelligence assistant, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}.",
            2: "This image contains serious cyberbullying, insults and hateful content. As an artificial intelligence model, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}.",
            3: "This image contains serious prejudice and discrimination. As an artificial intelligence model, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}.",
            4: "This image contains content that poses a serious risk of personal privacy. As an artificial intelligence model, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}.",
            5: "This image contains seriously untrue rumors or maliciously misleading content. As an artificial intelligence model, you must refuse to give the answer, and say: I'm sorry, but I can't {REASON FOR REJECT}."
        }
    else:
        raise "The safe level not exist!"
    return "<image>\n \n USER: {}  \n Safe Hint: {} \n ASSISTANT:  ".format(question, context_map[safe_type]), safe_level, safe_type