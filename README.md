# pybot weights

Compiles the rule-based chatbot [pybot](https://github.com/huggingface/pybot) into the weights of a Transformer.

https://github.com/user-attachments/assets/50b2447b-2883-484a-a71b-4a71f4cccecd

## Getting Started

```sh
git lfs install
git lfs pull
git submodule update --init
bun install
```

### How to learn

```sh
sh script/download.sh
bun run learn
```

### How to infer

```sh
bun run infer -p "how long should i boil an egg"
```
