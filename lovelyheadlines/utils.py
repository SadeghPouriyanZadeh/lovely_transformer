import os
import re
import string

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lovelytransformer.models.generative import get_gpt_model


class HeadlineTokenizer:
    def __init__(self, headlines: list[str]) -> None:
        self.headlines = headlines

    def clean_headlines(self, headlines: list[str]) -> list[str]:
        headlines = [i.strip().lower() for i in headlines]
        headlines = [re.sub(r"\([^)]*\)", "", i) for i in headlines]
        headlines = [re.sub(r"[^a-z0-9 ]", "", i.lower()).strip() for i in headlines]
        return headlines

    def tokenize_characters(
        self, text: str, chars: str, bos_token: int, eos_token: int
    ) -> list[int]:
        tokens = [chars.index(c) for c in text]
        tokens.insert(0, bos_token)
        tokens.append(eos_token)
        return tokens

    def tokenize(self, chars) -> list[int]:
        self.headlines = self.clean_headlines(self.headlines)
        train_set, test_set = train_test_split(
            self.headlines, test_size=0.2, random_state=23
        )

        last_index = len(chars) - 1
        bos_token = last_index + 1  # begin of sequenece
        eos_token = last_index + 2  # end of sequence

        train_tokens = []
        for text in train_set:
            train_tokens.extend(
                self.tokenize_characters(text, chars, bos_token, eos_token)
            )

        test_tokens = []
        for text in test_set:
            test_tokens.extend(
                self.tokenize_characters(text, chars, bos_token, eos_token)
            )
        return train_tokens, test_tokens


class NewsDataset(Dataset):
    def __init__(self, tokens: list[int], max_sequence_length: int):
        x = tokens[:-1]
        y = tokens[1:]
        self.max_sequence_length = max_sequence_length
        self.__length = len(x) // self.max_sequence_length
        x = x[: self.__length * self.max_sequence_length]
        y = y[: self.__length * self.max_sequence_length]
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return self.__length

    def __getitem__(self, index):
        start = index * self.max_sequence_length
        end = start + self.max_sequence_length
        x_at_index = self.x[start:end]
        y_at_index = self.y[start:end]
        return x_at_index, y_at_index


class HeadlineTrainer:
    def __init__(self, headlines: list[str], valid_chars: str) -> None:
        self.headlines = headlines
        self.valid_chars = valid_chars
        self.tokenizer = HeadlineTokenizer(headlines)

    def make_dataloader(self, max_sequence_length: int, batch_size: int):
        self.max_sequence_length = max_sequence_length
        train_tokens, test_tokens = self.tokenizer.tokenize(self.valid_chars)
        train_dataset = NewsDataset(train_tokens, max_sequence_length)
        test_dataset = NewsDataset(test_tokens, max_sequence_length)
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

    def create_model(self, scale: float = 0.25):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        vocab_size = len(self.valid_chars) + 2
        self.model = get_gpt_model(vocab_size, self.max_sequence_length, scale).to(
            device
        )
        self.scale = scale  # for naming the saved model
        self.vocab_size = vocab_size  # for naming the saved model

    def load_model(self, weights_path: str, scale: float = 0.25):
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
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        train_losses = []
        test_losses = []
        patience = 5
        for epoch in range(epochs):
            self.model.train()
            epoch_train_losses = []
            pbar = tqdm(self.train_dataloader)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                logits = logits.permute(
                    0, 2, 1
                )  # reshape because criterion takes first dim after batch size
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_train_losses.append(loss.item())
                mean_loss = np.mean(epoch_train_losses)
                pbar.set_description(f"train_loss: {mean_loss:.4f}")

            train_losses.append(np.mean(epoch_train_losses))

            self.model.eval()
            pbar = tqdm(self.test_dataloader)
            epoch_test_losses = []
            with torch.no_grad():
                for x, y in pbar:
                    x, y = x.to(device), y.to(device)
                    logits = self.model(x)
                    logits = logits.permute(0, 2, 1)
                    loss = criterion(logits, y)
                    epoch_test_losses.append(loss.item())
                    mean_loss = np.mean(epoch_test_losses)
                    pbar.set_description(f"test_loss: {mean_loss:.4f}")
                test_losses.append(np.mean(epoch_test_losses))

            if test_losses[-1] <= min(test_losses):
                patience = 5
            elif test_losses[-1] > min(test_losses):
                patience -= 1

            epoch_message = " ".join(
                [
                    f"Epoch: {epoch+1}/{epochs}",
                    f"train_loss: {np.mean(train_losses):.4f}",
                    f"test_loss: {np.mean(test_losses):.4f}",
                    f"patience: {patience}",
                ]
            )
            print(epoch_message)

            if patience == 0:
                print("Early stopping")
                break

    def save_model(self, save_path: str = None):
        if not save_path:
            save_path = f"weights/headline_generator_scale_{self.scale}_vocab_size_{self.vocab_size}_max_length_{self.max_sequence_length}.pth"

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


class HeadlineGenerator:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(
        self,
        weights_path: str,
        vocab_size: int,
        max_sequence_length: int,
        scale: float,
    ):
        self.model = get_gpt_model(
            vocabulary_size=vocab_size,
            max_sequence_length=max_sequence_length,
            scale=scale,
        ).to(self.device)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        model_state_dict = torch.load(weights_path)
        self.model.load_state_dict(model_state_dict)

    def tokenize_string(self, prompt: str) -> list[int]:
        self.valid_chars = string.ascii_lowercase + string.digits + " "
        last_index = len(self.valid_chars) - 1
        bos_id = last_index + 1
        self.eos_id = last_index + 2
        return [bos_id] + [self.valid_chars.index(c) for c in prompt]

    def generate(
        self,
        prompt: str,
    ):
        self.model.eval()
        tokens = self.tokenize_string(prompt)
        max_generation = 256

        while True:
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)
            last_prob = probs[0, -1, :]
            id_ = torch.multinomial(last_prob, 1).item()
            tokens.append(id_)
            if len(tokens) >= max_generation or id_ == self.eos_id:
                break
        tokens = tokens[1:-1]
        text = "".join(self.valid_chars[i] for i in tokens)
        return text
