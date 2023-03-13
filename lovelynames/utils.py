import os
import string

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from lovelytransformer.models.generative import get_gpt_model


class StringDataset(Dataset):
    def __init__(self, strings: list[str]) -> None:
        self.strings = strings
        characters = set()
        for n in strings:
            characters.update(n)
        characters = list(characters)
        characters.sort()
        self.characters = characters
        last_index = len(characters) - 1
        self.padding_index = last_index + 1
        self.vocab_size = len(characters) + 1
        self.max_length = max([len(n) for n in strings])
        print(f"Dataset vocabulary size: {self.vocab_size}")

    def tokenize_string(self, name: str) -> list[int]:
        return [self.characters.index(i) for i in name]

    def pad(self, tokenized_string: list[int]):
        tokens = tokenized_string.copy()
        for _ in range(self.max_length - len(tokens)):
            tokens.append(self.padding_index)
        return tokens

    def __len__(self):
        return len(self.strings)

    def __getitem__(self, index):
        string = self.strings[index]
        tokenized_string = self.tokenize_string(string)
        padded_tokenized_string = self.pad(tokenized_string)
        x = padded_tokenized_string[:-1]
        y = padded_tokenized_string[1:]
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class StringTrainer:
    def __init__(self, dataset: StringDataset) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Training will be on '{self.device}' device.")
        self.dataset = dataset

    def make_dataloaders(
        self,
        batch_size: int = 256,
        test_size: float = 0.2,
    ):
        test_length = round(test_size * len(self.dataset))
        train_length = len(self.dataset) - test_length
        train_dataset, test_dataset = random_split(
            self.dataset,
            [train_length, test_length],
            generator=torch.Generator().manual_seed(23),
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

    def create_model(self, scale: float = 0.25):
        self.scale = scale
        self.model = get_gpt_model(
            vocabulary_size=self.dataset.vocab_size,
            max_sequence_length=self.dataset.max_length,
            scale=scale,
        ).to(self.device)
        print(f"Model scale: {scale}")
        print(f"Dataset vocabulary size: {self.dataset.vocab_size}")
        print(f"Dataset max sequence length: {self.dataset.max_length}")

    def load_model(self, weights_path: str, scale: float = 0.25):
        self.scale = scale
        self.model = get_gpt_model(
            vocabulary_size=self.dataset.vocab_size,
            max_sequence_length=self.dataset.max_length,
            scale=scale,
        ).to(self.device)

        model_state_dict = torch.load(weights_path)
        self.model.load_state_dict(model_state_dict)

    def train(
        self,
        epochs: int,
        patience: int = 5,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        pbar = tqdm(range(epochs))
        train_losses = []
        test_losses = [float("inf")]
        for _ in pbar:
            self.model.train()
            for x, y in self.train_dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                logits = logits.permute(0, 2, 1)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                pbar.set_postfix(
                    {
                        "train_loss": np.mean(train_losses[-10:]),
                        "test_loss": test_losses[-1],
                    }
                )
            self.model.eval()
            with torch.no_grad():
                test_loss = 0
                for x, y in self.test_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    logits = self.model(x)
                    logits = logits.permute(0, 2, 1)
                    loss = self.criterion(logits, y)
                    test_loss += loss.item()
                test_loss /= len(self.test_dataloader)
            test_losses.append(test_loss)

            if patience:
                if test_losses[-1] <= min(test_losses):
                    patience = 5

                elif test_losses[-1] > min(test_losses):
                    print("Early Stopping...")
                    patience -= 1
                    print(f"Lowering patience to {patience}")

                if patience == 0:
                    print("Training is finished.")
                    break

    def save_model(self, save_path: str = None):
        save_path = f"weights/name_generator_scale_{self.scale}_vocab_size_{self.dataset.vocab_size}_max_length_{self.dataset.max_length}.pth"

        if os.path.exists(save_path):
            base_path, ext = os.path.splitext(save_path)
            i = 1
            while True:
                new_file_path = f"{base_path}_{i}{ext}"
                if not os.path.exists(new_file_path):
                    break
                i += 1
            save_path = new_file_path
            print("New save path:", new_file_path)

        torch.save(self.model.state_dict(), save_path)


class NameGenerator:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(
        self,
        weights_path: str,
        vocab_size: int,
        max_length: int,
        scale: float = 0.25,
    ):
        self.model = get_gpt_model(
            vocabulary_size=vocab_size,
            max_sequence_length=max_length,
            scale=scale,
        ).to(self.device)
        self.max_length = max_length
        self.vocab_size = vocab_size
        model_state_dict = torch.load(weights_path)
        self.model.load_state_dict(model_state_dict)

    def tokenize_string(self, prompt: str) -> list[int]:
        self.characters = string.ascii_lowercase
        return [self.characters.index(i) for i in prompt]

    def generate(self, prompt):
        self.model.eval()
        ids = self.tokenize_string(prompt)
        last_index = len(self.characters) - 1
        self.padding_index = last_index + 1

        while True:
            if (len(ids) >= self.max_length) or (ids[-1] == self.padding_index):
                break
            ids_tensor = (
                torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
            )
            logits = self.model(ids_tensor)
            probs = logits.softmax(dim=-1)[0, -1, :]
            ids.append(torch.multinomial(probs, 1).item())
        return "".join([self.characters[i] for i in ids if i != self.padding_index])
