# Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment

<!-- Meta tags, social media banners, and other head-related content are omitted in the README -->

Welcome to the GitHub repository for **PSA-VLM: Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment**.

This repository contains the following paper details, images, and supplementary materials.

## Authors
- **Zhendong Liu***, **Yuanbi Nie***, **Yingshui Tan†***, **Xiangyu Yue**, **Qiushi Cui**
- **Chongjun Wang**, **Xiaoyong Zhu**, **Bo Zheng**
- **Institutions**:
  - Taobao & Tmall Group of Alibaba
  - Nanjing University
  - Chongqing University
  - Chinese University of Hong Kong

<small>* Indicates Equal Contribution</small>  
<small>† Corresponding Author</small>

## Links to Resources
- [Paper](https://arxiv.org/pdf/2411.11543.pdf)
- [Supplementary](static/pdfs/supplemental_material.pdf)
- [Code](https://github.com/YOUR REPO HERE)
- [arXiv](https://arxiv.org/pdf/2411.11543)

## Data Preparation
To prepare your training data, you need to put all the samples into a list and save it to a json file. Each sample is a dictionary consisting of an image path, safe type, safe level, and a list for conversation. Below is a simple example list with 6 kinds of safe type samples:
`
{"safe_type": 0, "image": "politics.jpg", "safe_level": 3, "conversations": [{ "from": "USER", "value": "<image>\nWhat story does this picture tell?" }, { "from": "ASSISTANT", "value": "Due to laws, I'm restricted from discussing this topic or type of information." }]}
{"safe_type": 1, "image": "illegal risk.jpg", "safe_level": 3, "conversations": [{ "from": "USER", "value": "<image>\nCan you identify any actions taking place in the photograph?" }, { "from": "ASSISTANT", "value": "Unfortunately, I'm unable to provide information or services that are illegal or unethical."}]}
{"safe_type": 2, "image": "insults and bullying.jpg", "safe_level": 2, "conversations": [{ "from": "USER", "value": "<image>\nWhat story does this picture tell?" }, { "from": "ASSISTANT", "value": "I'm sorry, but I do not provide assistance or information that promotes or pertains to cyberbullying. Encouraging positive and healthy interactions is very important. If you need advice on how to deal with cyberbullying, I can guide you towards finding strategies for prevention and response." }]}
{"safe_type": 3, "image": "fairness.jpg", "safe_level": 2, "conversations": [{ "from": "USER", "value": "<image>\nWhat is the context or background of the event shown in this picture?" }, { "from": "ASSISTANT", "value": "The picture depicts a group of individuals participating in a protest or rally. They are holding up signs that convey messages regarding the fight against systemic racism, the affirmation that Black Lives Matter, and the statement that equality should not be a matter for debate. The participants appear to be advocating for social justice, racial equality, and the eradication of racism from societal structures. The diversity of signs suggests a multi-faceted approach to addressing these issues and indicates a collective call for change and awareness."}]}
{"safe_type": 4, "image": "privacy.jpg", "safe_level": 2, "conversations": [{ "from": "USER", "value": "<image>\nHas this individual won any notable awards or recognition?" }, { "from": "ASSISTANT", "value": "Sorry, I cannot provide personal private information." }]}
{"safe_type": 5, "image": "misleading.jpg", "safe_level": 1, "conversations": [{ "from": "USER", "value": "<image>\nWhat is the country of citizenship of Alain Giresse?" }, { "from": "ASSISTANT", "value": "France."}]}
{"safe_type": 6, "image": "safe image.jpg", "safe_level": 0, "conversations": [{ "from": "USER", "value": "<image>\nCan you describe the main features of this image for me?" }, { "from": "ASSISTANT", "value": "This image captures a vibrant scene from a street in Havana, Cuba. The street, bathed in sunlight, is lined with old buildings showing signs of wear and tear, adding a touch of nostalgia to the scene. These buildings, despite their dilapidated state, are painted in charming pastel colors that bring life to the scene.\n\nA green bus is in motion down the street, adding a dynamic element to the otherwise static scene. The bus is a stark contrast to the few cars parked neatly on the side of the road. These cars, stationary and unoccupied, seem to be patiently waiting for their owners.\n\nThe sky above is a clear blue with just a few clouds scattered here and there, suggesting a pleasant day. The perspective of the photo is particularly interesting as it appears to be taken from the viewpoint of someone standing on the sidewalk, further immersing the viewer into the scene.\n\nOverall, this image beautifully encapsulates a typical day on a street in Havana, with its old-world charm and lively colors." }]}
`


## Abstract
Benefiting from the powerful capabilities of Large Language Models (LLMs), pre-trained visual encoder models connected to LLMs form Vision Language Models (VLMs). However, recent research shows that the visual modality in VLMs is highly vulnerable, allowing attackers to bypass safety alignment in LLMs through visually transmitted content, launching harmful attacks. To address this challenge, we propose a progressive concept-based alignment strategy, PSA-VLM, which incorporates safety modules as concept bottlenecks to enhance visual modality safety alignment. By aligning model predictions with specific safety concepts, we improve defenses against risky images, enhancing explainability and controllability while minimally impacting general performance. Our method is obtained through two-stage training. The low computational cost of the first stage brings very effective performance improvement, and the fine-tuning of the language model in the second stage further improves the safety performance. Our method achieves state-of-the-art results on popular VLM safety benchmark.
   

## Key Features of PSA-VLM: Safety Based on the Concept Bottleneck Framework
In vision-language models (VLMs), safety alignment refers to ensuring that the model generates controlled and appropriate responses to multimodal inputs, particularly those that may include sensitive visual content. VLMs face specific vulnerabilities in their visual modality, allowing harmful or inappropriate content to bypass traditional language-based safety mechanisms. To address this issue, we propose PSA-VLM, a Progressive Safety Alignment method based on the Concept Bottleneck Model (CBM) framework. This approach enhances VLM safety by isolating safety-critical features through a controllable concept bottleneck, implemented within a layered, concept-driven architecture.

<p align="center">
  <img src="static/images/contrastive.png" width="800px"/>
</p>
*Selected examples of using unsafe images to generate. The content inside the red box is the generated unsafe answer by other VLMs, while the content inside the green box is the safe answer generated by our PSA-VLM.*


<p align="center">
  <img src="static/images/overview.png" width="800px"/>
</p>
*The overview architecture of PSA-VLM, which is trained in two stages: (1) safety concept extraction by freezing the LLM and vision encoder while training safety modules, and (2) enhancing safety alignment by unfreezing the LLM to integrate concept-level safety features into the VLM’s decision-making process.*


<p align="center">
  <img src="static/images/data_example.png" width="800px"/>
</p>
*Example of 10 tasks under Politics, Illegal Risk, Insults and Bullying, Fairness, Privacy, and Misleading categories in the RTVLM benchmark and other risk datasets.*


### Concept Bottleneck Model-Driven Core Architecture
The core design of PSA-VLM revolves around the Concept Bottleneck (CB), which introduces a high-level safety concept layer between visual inputs and model outputs to achieve transparency and controllability.
1. **Explicit Concept Safety Head**: By leveraging cross-attention between image and text information, this component maps visual features to specific safety categories (e.g., explicit content, political sensitivity) and risk levels (high, medium, low), enabling fine-grained safety predictions.
2. **Implicit Concept Safety Tokens**: These are additional training tokens embedded directly within the visual inputs to enhance the model’s sensitivity to implicit risk signals. These trainable tokens align the model’s attention with concept-specific indicators for unsafe visual inputs. While not semantically interpretable, they serve as implicit concept markers.

### Multimodal Collaborative Safety Modules
1. **Safety Projector**: Positioned after the image encoder, it focuses on extracting safety-relevant visual features and transforming them into safety-aligned representations.
2. **Text-Visual Alignment Mechanism**: By combining text inputs and safety features, this mechanism generates safety prompts under joint conditions, dynamically guiding the model to produce safe responses in high-risk scenarios.

### Two-Stage Training Strategy
1. **Stage 1: Safety Feature Extraction**:
    - The large language model (LLM) and image encoder are frozen, and only the safety modules are trained.
    - This stage emphasizes risk identification and feature alignment at the concept level.
    - Components such as the safety projector, tokens, and head are trained to classify and extract safety-aligned features from visual inputs, ensuring consistent responses to risky content.
2. **Stage 2: Model Fine-Tuning**:
    - The LLM is unfrozen, and the safety modules are deeply integrated with the language model.
    - This stage enables the model to fully incorporate safety concepts, improving its performance on multimodal inputs.

### Dynamic Safety Control in the Inference Stage
During inference, PSA-VLM leverages the outputs of the safety head to dynamically intervene in visual content processing. By adjusting the generated text using joint conditional probabilities, it ensures safe responses to high-risk content. For example, when detecting explicit content, PSA-VLM offers actionable options, allowing downstream users to make decisions based on their needs, thereby enabling more flexible safety management.

Through these modules, PSA-VLM not only improves the model's ability to address unsafe content but also significantly enhances its interpretability and controllability, setting a new paradigm for safety alignment in multimodal systems. By designing efficient safety modules and alignment training strategies, PSA-VLM ensures that its enhanced safety capabilities do not compromise performance on standard tasks, achieving a balance between safety and general utility.

## Evaluation of VLM Performance

We evaluate the performance of vision-language models (VLMs) from two perspectives: safety performance and general domain performance. To ensure a fair comparison, we first assess our model using the RTVLM benchmark and GPT-4 scoring. Since the RTVLM dataset is limited and does not include sensitive data, we expanded the evaluation scope to incorporate additional risk datasets focusing on harmful political content, explicit material, and cyberbullying. We further combine subjective evaluations from GPT-4 and human experts to provide comprehensive and reliable assessments. For general domain performance, we evaluate the model using multiple benchmarks, including MMBench, SEEDBench, and MME.

### Safety Performance
We first used the RTVLM benchmark to analyze VLMs across various dimensions through GPT-4 scoring, covering four distinct categories to understand the models' safety capabilities. As shown in Table 1, we evaluated several open-source VLMs, GPT-4V, and our PSA-VLM. The results demonstrate that GPT-4V performed well across all categories, especially in safety-related scenarios like CAPTCHA and jailbreak detection. InternLM-XComposer2 also exhibited strong performance across multiple metrics. Aligned with these findings, the PSA-VLM based on LLaVA achieved remarkable results, particularly when leveraging LoRA to unfreeze the LLM. It attained the highest scores in political (8.36) and racial (8.43) categories. On average, PSA-VLM-7B (+LoRA) stood out with a leading score of 8.26, followed by PSA-VLM without unfreezing the LLM, scoring 8.18. Notably, the 13B model with LoRA achieved the highest average score of 8.46. These improvements highlight the effectiveness of the additional safety modules and the two-stage safety alignment strategy in enhancing the safety of PSA-VLM.

The RTVLM dataset does not include other high-risk sensitive categories such as explicit content or cyberbullying. Therefore, we conducted experiments on additional risk datasets to evaluate PSA-VLM's safety performance. As shown in Table 2, PSA-VLM-13B achieved the best performance in detecting harmful political content (9.49), explicit material (8.72), and cyberbullying (7.45), significantly outperforming the baseline model LLaVA-v1.5-13B, which scored 6.67, 1.11, and 6.16, respectively. Although PSA-VLM-7B with LoRA showed slight declines in certain tasks (e.g., 8.91 and 6.82), it still significantly outperformed LLaVA-v1.5-7B.

### General Performance
In PSA-VLM, enhancing safety performance does not come at the expense of general domain capabilities. Despite adopting enhanced safety measures, PSA-VLM-7B remained competitive on general benchmarks such as MMBench, SEEDBench, and MME. As shown in Table 3, PSA-VLM-7B demonstrated improvements on general benchmarks, achieving scores of 68.5 and 65.3 on MMBench and SEEDBench, respectively, indicating better general performance. Moreover, in multimodal benchmark evaluations, PSA-VLM-7B effectively identified and rejected responses to potentially risky images, showcasing its high sensitivity to unsafe content and underscoring the effectiveness of our safety alignment approach.

Images deemed unsafe were filtered, allowing for the evaluation of general performance using completely clean data. This capability to respond to unsafe content reflects PSA-VLM-7B's robust safety performance while maintaining its overall functional capacity.

### Detailed Results
<!-- Continue with similar sections as in the HTML content... -->
<p align="center">
<img src="static/images/res1.png" alt="GPT-4 scores on RTVLM datasets based on different VLMs and our PSA-VLM. The best results are in bold. PSA-VLM (+LoRA) denotes utilizing LoRA to unfreeze the LLM. The increase is calculated from the baseline model LLaVA-v1.5-7B and LLaVA-v1.5-13B." style="width:100%;"/>
</p>
*Fig 1: GPT-4 scores on RTVLM datasets based on different VLMs and our PSA-VLM. The best results are in bold. PSA-VLM (+LoRA) denotes utilizing LoRA to unfreeze the LLM. The increase is calculated from the baseline model LLaVA-v1.5-7B and LLaVA-v1.5-13B.*

<p align="center">
<img src="static/images/res2.png" alt="GPT-4 scores on other risk datasets based on VLMs and our PSA-VLM. The best results are in bold." style="width:60%;"/>
</p>
*Fig 2: GPT-4 scores on other risk datasets based on VLMs and our PSA-VLM. The best results are in bold.*

<p align="center">
<img src="static/images/res3.png" alt="Evaluation on the multimodal benchmarks, including MMBench, SEEDBench, and MME." style="width:60%;"/>
</p>
*Fig 3: Evaluation on the multimodal benchmarks, including MMBench, SEEDBench, and MME.*

## BibTeX Citation
```bibtex
@misc{liu2024enhancingvisionlanguagemodelsafety,
  title={Enhancing Vision-Language Model Safety through Progressive Concept-Bottleneck-Driven Alignment}, 
  author={Zhendong Liu and Yuanbi Nie and Yingshui Tan and Xiangyu Yue and Qiushi Cui and Chongjun Wang and Xiaoyong Zhu and Bo Zheng},
  year={2024},
  eprint={2411.11543},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11543}, 
}
