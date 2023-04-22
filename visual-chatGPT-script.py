# code adapted from https://github.com/microsoft/TaskMatrix/blob/main/visual_chatgpt.py
import csv
import time

from visual_chatgpt import *

OUTPUT = "/home/shivani/work/vilt/vsgpt_output.tsv"
INPUT = "/home/shivani/work/vilt/new_test.tsv"

class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        
        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(temperature=0)
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang = "English"):
        self.memory.clear() #clear previous history
        if lang=='English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )

    def run_text(self, text, state = []):
        self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        return state

    def run_image(self, image, state = []):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print(image_filename)
        print("======>Auto Resize Image...")
        img = Image.open(image)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        print("description: , ", description)
        Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
        AI_prompt = "Received.  "
        self.agent.memory.buffer = self.agent.memory.buffer + Human_prompt + 'AI: ' + AI_prompt
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        # print("state: ", state)
        return description, state
    
load = "Text2Box_cuda:0,Segmenting_cuda:0,Inpainting_cuda:0,ImageCaptioning_cuda:0"

load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in load.split(',')}
bot = ConversationBot(load_dict=load_dict)
bot.init_agent("English")

with open(OUTPUT, 'w') as output:
    with open(INPUT, 'r') as input:
        lines = csv.reader(input, delimiter="\t")
        writer = csv.writer(output, delimiter="\t")
        for i, row in enumerate(lines):
            image = "image" + str(row[2]) + ".png" 
            image = "/home/shivani/work/data/images/" + image
            # prompt = "along with this image image/ddb21a00.png . Now list all the food items present. image/ddb21a00.png"
            output, state = bot.run_image(image)
            prompt = "the user is mentioning " + row[0] + " and the image description "+ image + ", give all the food items that the user ate in the form of a list (ex a bowl of food and glass of milk as ['food', 'milk']), give only the food items with no jargons" 
            state = bot.run_text(prompt, state)
            output = state[1][1]
            bot.memory.clear()
            writer.writerow([output])
            if i % 40 == 0:
                time.sleep(200)
