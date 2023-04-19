import asyncio
import json
import csv
import time

from EdgeGPT import Chatbot, ConversationStyle
from pathlib import Path
from pprint import pprint


# bot = Chatbot(cookiePath='./cookie.json')

# with open('./cookie.json', 'r') as f:
#     cookies = json.load(f)
# bot = Chatbot(cookies=cookies)

path = Path("/home/sg/work/scrapy_selenium")
COOKIE_PATH = "/home/sg/work/scrapy_selenium/cookie.json"

async def main():
    bot = Chatbot(cookiePath=COOKIE_PATH)
    with open(path / "final_smoothies_categories.tsv", "r") as input:
        with open(path / "final_smoothies_sentences1.tsv", "w") as output:
            tsv_reader = csv.reader(input, delimiter="\t")
            tsv_writer = csv.writer(output, delimiter="\t")
            for i, line in enumerate(tsv_reader):
                line = line[0]
                prompt = line + " is a name of a specific food item now generate a sentences how it could be logged into a diet tracking log book eg: I ate " + line + " Today. Give me only the sentence, with no jargons"
                sen = await bot.ask(prompt= prompt, conversation_style=ConversationStyle.creative)
                item = sen['item']
                message = item['messages'][1]
                text = message['text']
                tsv_writer.writerow([text])
                time.sleep(2)
                if i == 15:
                    break
        await bot.close()


if __name__ == "__main__":
    asyncio.run(main())
