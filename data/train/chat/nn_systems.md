### User:
What neural network does local chat use?
### Assistant:
Local chat uses the custom text transformer saved under the text enterprise or text checkpoint folder, not an external cloud model API.

### User:
Where are model weights stored?
### Assistant:
Weights live in checkpoints folders such as text_enterprise latest.pt, image latest.pt, and video latest.pt, with matching tokenizer.json for text.

### User:
What training types exist?
### Assistant:
You can train general, chat, coding, creative, math, multimodal-text, games, nsfw, unrestricted, and voice, plus full image and video pipelines.

### User:
How do I fine-tune chat?
### Assistant:
Add dialogue blocks under data train chat as txt or md, then run train chat. Optional steps flag controls length.

### User:
How does coding training work?
### Assistant:
Coding rows are JSONL with language, prompt, and code. train coding finetunes the shared text network on those samples.

### User:
What is multimodal text training?
### Assistant:
It finetunes the text network on media captions so image and video descriptions align with generation prompts.

### User:
How do games training and observe connect?
### Assistant:
observe captures screen frames into data train games sessions and log dialogue. train games absorbs strategy text and session summaries.

### User:
Is voice a separate neural net?
### Assistant:
Speech uses STT and TTS engines. Language understanding still comes from the text network and the assistant intent layer.

### User:
How do I list train targets?
### Assistant:
Run train list from the CLI to print each training type and its data format.
