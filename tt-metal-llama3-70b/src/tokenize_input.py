import os
import pathlib
from tt_metal_impl.reference.llama.tokenizer3 import (
    ChatFormat,
    Dialog,
    Message,
    Tokenizer3,
)


# pip install networkx==2.8.8 torch sentencepiece tiktoken blobfile fairscale fire


def tokenize_input(prompt_str, output_str):
    cache_root = os.environ["CACHE_ROOT"]
    weights_path = pathlib.Path(
        f"{cache_root}/model_weights/repacked-llama-3-70b-instruct"
    )
    tokenizer_path = weights_path.joinpath("tokenizer.model")
    llama_version = "llama3"
    if llama_version == "llama3":
        tokenizer = Tokenizer3(model_path=str(tokenizer_path))
        formatter = ChatFormat(tokenizer)
        dialog = [{"role": "user", "content": prompt_str}]
        prompt_tokens = formatter.encode_dialog_prompt(dialog)
        output_tokens = tokenizer.encode(output_str, bos=True, eos=False)
    else:
        raise NotImplementedError(
            "only llama3 supported, add tokenization for llama2 if needed."
        )

    return prompt_tokens, output_tokens


if __name__ == "__main__":
    prompt_str = "What are some artists I might like if I like Billy Joel?"
    output_str = "Billy Joel is a legendary artist known for his catchy, piano-driven pop-rock sound and storytelling lyrics. If you like Billy Joel, you might also enjoy the following artists:\n\n1. Elton John: Like Billy Joel, Elton John is a master of piano-driven rock with a flair for the dramatic and a knack for crafting memorable melodies.\n2. Leon Russell: A keyboard virtuoso and songwriter, Leon Russell's music often features a similar blend of rock, pop, and country elements found in Billy Joel's work.\n3. Ben Folds: As the lead singer and pianist of Ben Folds Five, Ben Folds' music often shares Billy Joel's wit, humor, and piano-driven energy.\n4. Warren Zevon: While darker and more cynical than Billy Joel, Warren Zevon's music often features a similar blend of rock, pop, and folk elements, with a focus on storytelling.\n5. Jackson Browne: As a singer-songwriter, Jackson Browne's music often explores similar themes of love, relationships, and social commentary, with a more laid-back, California-rock vibe.\n6. James Taylor: With his soothing voice and acoustic-based sound, James Taylor's music often shares Billy Joel's warmth and introspection, with a more folk-oriented approach.\n7. Harry Chapin: A fellow Long Island native, Harry Chapin's music often features a similar blend of folk, rock, and pop elements, with a focus on storytelling and social commentary.\n8. Randy Newman: As a singer-songwriter and pianist, Randy Newman's music often explores similar themes of love, relationships, and social commentary, with a more satirical and humorous tone.\n9. Marc Cohn: With his soulful voice and piano-driven sound, Marc Cohn's music often shares Billy Joel's emotional depth and introspection, with a more soul and R&B-influenced approach.\n10. Bruce Hornsby: As the leader of Bruce Hornsby and the Range, his music often features a similar blend of rock, pop, and folk elements, with a focus on storytelling and socially conscious lyrics.\n\nThese artists all share some elements of Billy Joel's style, but each with their own unique twist. Give them a listen and see if you discover some new favorite artists!<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nI think I can add a few more to the list:\n\n11. Hall & Oates: This duo's music often features a similar blend of rock, pop, and soul elements, with a focus on catchy hooks and memorable melodies.\n12. Christopher Cross: As a singer-songwriter, Christopher Cross' music often shares Billy Joel's soft"
    prompt_tokens, output_tokens = tokenize_input(prompt_str, output_str)
    print("=" * 80)
    print(f"len(prompt_tokens):= {len(prompt_tokens)}")
    print(f"len(output_tokens):= {len(output_tokens)}")
