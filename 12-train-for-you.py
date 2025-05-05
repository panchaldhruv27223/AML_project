import os
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import psutil
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from scipy.sparse.csgraph import minimum_spanning_tree
import shutil
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s: %(message)s")

class GraphDependencyParser(nn.Module):
    def __init__(self, word_vocab_size, pos_vocab_size, audio_dim=768, hidden_dim=256, num_relations=50):
        super(GraphDependencyParser, self).__init__()
        self.word_embed = nn.Embedding(word_vocab_size, 100)
        self.pos_embed = nn.Embedding(pos_vocab_size, 50)
        self.audio_proj = nn.Linear(audio_dim, 100)
        self.mlp = nn.Sequential(
            nn.Linear(250, hidden_dim),  # 100 (word) + 50 (pos) + 100 (audio)
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.arc_scorer = nn.Linear(hidden_dim, hidden_dim)
        self.rel_scorer = nn.Linear(hidden_dim * 2, num_relations)
        self.dropout = nn.Dropout(0.3)

    def forward(self, words, pos, audio_feats, start_times, end_times):
        word_emb = self.word_embed(words)  # [batch, seq_len, 100]
        pos_emb = self.pos_embed(pos)      # [batch, seq_len, 50]
        audio_emb = self.align_audio(audio_feats, start_times, end_times)  # [batch, seq_len, 100]
        combined = torch.cat([word_emb, pos_emb, audio_emb], dim=-1)  # [batch, seq_len, 250]
        combined = self.mlp(combined)  # [batch, seq_len, hidden_dim]
        combined = self.dropout(combined)
        head_feats = self.arc_scorer(combined)  # [batch, seq_len, hidden_dim]
        dep_feats = combined  # [batch, seq_len, hidden_dim]
        arc_scores = torch.einsum("bih,bjh->bij", head_feats, dep_feats)  # [batch, seq_len, seq_len]
        head_dep = torch.cat([
            head_feats.unsqueeze(2).repeat(1, 1, head_feats.size(1), 1),
            dep_feats.unsqueeze(1).repeat(1, dep_feats.size(1), 1, 1)
        ], dim=-1)  # [batch, seq_len, seq_len, 2*hidden_dim]
        rel_scores = self.rel_scorer(head_dep)  # [batch, seq_len, seq_len, num_relations]
        return arc_scores, rel_scores

    def align_audio(self, audio_feats, start_times, end_times):
        batch_size, seq_len = start_times.size()
        aligned = torch.zeros(batch_size, seq_len, audio_feats.size(-1), device=audio_feats.device)
        for b in range(batch_size):
            for i in range(seq_len):
                start = int(start_times[b, i] * 50)  # Assume 50 frames/sec
                end = int(end_times[b, i] * 50)
                if end > start and end <= audio_feats.size(1):
                    aligned[b, i] = audio_feats[b, start:end].mean(dim=0)
        return self.audio_proj(aligned)

def eisner_decode(arc_scores):
    batch_size, seq_len, _ = arc_scores.size()
    trees = []
    for b in range(batch_size):
        adj_matrix = -arc_scores[b].detach().cpu().numpy()
        mst = minimum_spanning_tree(adj_matrix).toarray()
        tree = np.argmax(-mst, axis=1)  # Convert to head indices
        trees.append(torch.tensor(tree, dtype=torch.long, device=arc_scores.device))
    return torch.stack(trees)  # [batch, seq_len]

class DependencyDataset(Dataset):
    def __init__(self, csv_path, word_vocab, pos_vocab, rel_vocab):
        self.df = pd.read_csv(csv_path)
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.rel_vocab = rel_vocab
        self.sample_rate = 16000

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {
            "ID": row["ID"],
            "duration": float(row["duration"]),
            "wav_path": row["wav"]
        }
        wav, sr = torchaudio.load(row["wav"])
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        item["wav"] = wav.squeeze(0)
        words = row["wrd"].split()
        pos = row["pos"].split()
        item["words"] = torch.tensor([self.word_vocab.get(w, 0) for w in words], dtype=torch.long)
        item["pos"] = torch.tensor([self.pos_vocab.get(p, 0) for p in pos], dtype=torch.long)
        heads = list(map(int, row["gov"].split()))
        rels = row["dep"].split()
        # Fix 1-based indexing and validate heads
        seq_len = len(words)
        valid_heads = []
        for h in heads:
            # Convert 1-based to 0-based
            h_adj = h - 1 if h > 0 else h
            if h_adj == -1 or 0 <= h_adj < seq_len:
                valid_heads.append(h_adj)
            else:
                logger.warning(f"Invalid head index {h} (adjusted {h_adj}) for ID {row['ID']}, seq_len {seq_len}. Row: {row.to_dict()}")
                valid_heads.append(0)  # Default to root
        item["heads"] = torch.tensor(valid_heads, dtype=torch.long)
        item["rels"] = torch.tensor([self.rel_vocab.get(r, 0) for r in rels], dtype=torch.long)
        start_times = list(map(float, row["start_word"].split()))
        end_times = list(map(float, row["end_word"].split()))
        item["start_times"] = torch.tensor(start_times, dtype=torch.float)
        item["end_times"] = torch.tensor(end_times, dtype=torch.float)
        return item

def custom_collate_fn(batch):
    output = {}
    max_len = max(len(item["words"]) for item in batch)
    output["ID"] = [item["ID"] for item in batch]
    output["wav"] = torch.nn.utils.rnn.pad_sequence(
        [item["wav"] for item in batch], batch_first=True
    )
    output["wav_lengths"] = torch.tensor([item["wav"].size(0) / 16000 for item in batch], dtype=torch.float)
    output["words"] = torch.nn.utils.rnn.pad_sequence(
        [item["words"] for item in batch], batch_first=True, padding_value=0
    )
    output["pos"] = torch.nn.utils.rnn.pad_sequence(
        [item["pos"] for item in batch], batch_first=True, padding_value=0
    )
    output["heads"] = torch.nn.utils.rnn.pad_sequence(
        [item["heads"] for item in batch], batch_first=True, padding_value=-1
    )
    output["rels"] = torch.nn.utils.rnn.pad_sequence(
        [item["rels"] for item in batch], batch_first=True, padding_value=0
    )
    output["start_times"] = torch.nn.utils.rnn.pad_sequence(
        [item["start_times"] for item in batch], batch_first=True, padding_value=0.0
    )
    output["end_times"] = torch.nn.utils.rnn.pad_sequence(
        [item["end_times"] for item in batch], batch_first=True, padding_value=0.0
    )
    return output

def compute_metrics(head_pred, rel_pred, head_true, rel_true):
    mask = (head_true != -1) & (head_true < head_pred.size(1))  # Valid heads within seq_len
    logger.debug(f"Mask shape: {mask.shape}, Head pred shape: {head_pred.shape}, Rel pred shape: {rel_pred.shape}")
    if mask.sum() == 0:
        return 0.0, 0.0  # Avoid division by zero
    uas = (head_pred[mask] == head_true[mask]).float().mean().item()
    las = ((head_pred[mask] == head_true[mask]) & (rel_pred[mask] == rel_true[mask])).float().mean().item()
    return uas, las

def compute_loss(arc_scores, rel_scores, heads, rels, arc_loss_fn, rel_loss_fn, device):
    batch_size, seq_len = heads.size()
    arc_loss = arc_loss_fn(arc_scores.transpose(1, 2), heads)
    rel_scores_selected = torch.zeros(batch_size, seq_len, rel_scores.size(-1), device=device)
    for b in range(batch_size):
        for i in range(seq_len):
            if heads[b, i] != -1:  # Valid head
                rel_scores_selected[b, i] = rel_scores[b, i, heads[b, i]]
    rel_scores_flat = rel_scores_selected.view(-1, rel_scores.size(-1))
    rels_flat = rels.view(-1)
    mask = rels_flat != 0
    if mask.sum() > 0:
        rel_loss = rel_loss_fn(rel_scores_flat[mask], rels_flat[mask])
    else:
        rel_loss = torch.tensor(0.0, device=device)
    return arc_loss + rel_loss

def write_conllu(batch_ids, words, pos, head_pred, rel_pred, word_vocab, pos_vocab, rel_vocab, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    inv_word_vocab = {v: k for k, v in word_vocab.items()}
    inv_pos_vocab = {v: k for k, v in pos_vocab.items()}
    inv_rel_vocab = {v: k for k, v in rel_vocab.items()}
    for sent_id, sent_words, sent_pos, sent_heads, sent_rels in zip(
        batch_ids, words, pos, head_pred, rel_pred
    ):
        output_file = os.path.join(output_dir, f"{split_name}_{sent_id}.conllu")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# text = {' '.join(inv_word_vocab.get(w.item(), '<unk>') for w in sent_words)}\n")
            f.write(f"# sent_id = {sent_id}\n")
            for j, (word, pos_tag, head, rel) in enumerate(zip(sent_words, sent_pos, sent_heads, sent_rels), 1):
                word_str = inv_word_vocab.get(word.item(), '<unk>')
                pos_str = inv_pos_vocab.get(pos_tag.item(), '<unk>')
                rel_str = inv_rel_vocab.get(rel.item(), 'dep')
                head_idx = head.item() if head.item() < len(sent_words) else 0
                f.write(f"{j}\t{word_str}\t{word_str}\t{pos_str}\t_\t_\t{head_idx}\t{rel_str}\t_\t_\n")
            f.write("\n")
        logger.info(f"Wrote CoNLL-U file: {output_file}")

def save_metrics(epoch, train_uas, train_las, test_uas, test_las, train_loss, test_loss, metrics_file):
    metrics_dir = os.path.dirname(metrics_file)
    os.makedirs(metrics_dir, exist_ok=True)
    data = {
        "epoch": [epoch],
        "train_uas": [train_uas],
        "train_las": [train_las],
        "test_uas": [test_uas],
        "test_las": [test_las],
        "train_loss": [train_loss],
        "test_loss": [test_loss]
    }
    df = pd.DataFrame(data)
    if not os.path.exists(metrics_file):
        df.to_csv(metrics_file, index=False)
    else:
        df.to_csv(metrics_file, mode='a', header=False, index=False)
    logger.info(f"Saved metrics to {metrics_file}")

def train_model():
    base_path = "D:\\Manish Prajapati\\AML Sounding Trees\\Project\\Growing_tree_on_sound\\graph_based"
    data_path = "D:\\Manish Prajapati\\LibriSpeech\\csv"
    checkpoint_dir = os.path.join(base_path, "model long")
    plot_dir = os.path.join(base_path, "plots")
    conllu_dir = os.path.join(base_path, "conllu")
    metrics_file = os.path.join(base_path, "metrics.csv")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(conllu_dir, exist_ok=True)

    logger.info(f"CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}")

    word_vocab = {"<pad>": 0, "<unk>": 1}
    pos_vocab = {"<pad>": 0, "<unk>": 1}
    rel_vocab = {"<pad>": 0, "<unk>": 1}
    for split in ["train_split.csv", "test_split.csv"]:
        df = pd.read_csv(os.path.join(data_path, split))
        for _, row in df.iterrows():
            words = row["wrd"].split()
            pos = row["pos"].split()
            rels = row["dep"].split()
            for w in words:
                if w not in word_vocab:
                    word_vocab[w] = len(word_vocab)
            for p in pos:
                if p not in pos_vocab:
                    pos_vocab[p] = len(pos_vocab)
            for r in rels:
                if r not in rel_vocab:
                    rel_vocab[r] = len(rel_vocab)

    train_dataset = DependencyDataset(
        os.path.join(data_path, "train_split.csv"), word_vocab, pos_vocab, rel_vocab
    )
    test_dataset = DependencyDataset(
        os.path.join(data_path, "test_split.csv"), word_vocab, pos_vocab, rel_vocab
    )

    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False, collate_fn=custom_collate_fn, num_workers=2
    )

    logger.info(f"Train dataset size: {len(train_dataset)} sentences (expected 4242)")
    logger.info(f"Test dataset size: {len(test_dataset)} sentences (expected 500)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h",
        do_normalize=True,
        return_attention_mask=True,
        padding=True
    )
    wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec2.eval()
    for param in wav2vec2.parameters():
        param.requires_grad = False
    model = GraphDependencyParser(
        word_vocab_size=len(word_vocab),
        pos_vocab_size=len(pos_vocab),
        num_relations=len(rel_vocab)
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    arc_loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    rel_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    epochs = 14
    best_test_uas = 0.0
    previous_checkpoint = None
    train_uas_scores, train_las_scores, train_losses = [], [], []
    test_uas_scores, test_las_scores, test_losses = [], [], []

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs", total=epochs):
        model.train()
        train_uas, train_las, train_epoch_losses = [], [], []
        logger.info(f"Epoch {epoch} - Training")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", total=len(train_loader)):
            optimizer.zero_grad()
            wav = batch["wav"].to(device)
            words = batch["words"].to(device)
            pos = batch["pos"].to(device)
            heads = batch["heads"].to(device)
            rels = batch["rels"].to(device)
            start_times = batch["start_times"].to(device)
            end_times = batch["end_times"].to(device)

            with torch.no_grad():
                wav_list = [wav[i].cpu().numpy() for i in range(wav.size(0))]
                inputs = processor(
                    wav_list,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                audio_feats = wav2vec2(**inputs).last_hidden_state

            arc_scores, rel_scores = model(words, pos, audio_feats, start_times, end_times)
            head_pred = eisner_decode(arc_scores)
            batch_size, seq_len = heads.size()
            rel_pred = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
            for b in range(batch_size):
                for i in range(seq_len):
                    if head_pred[b, i] < seq_len:
                        rel_pred[b, i] = rel_scores[b, i, head_pred[b, i]].argmax()

            loss = compute_loss(arc_scores, rel_scores, heads, rels, arc_loss_fn, rel_loss_fn, device)
            loss.backward()
            optimizer.step()

            uas, las = compute_metrics(head_pred, rel_pred, heads, rels)
            train_uas.append(uas)
            train_las.append(las)
            train_epoch_losses.append(loss.item())

            if device.type == "cuda":
                logger.debug(f"VRAM allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")

        train_uas_score = np.mean(train_uas) if train_uas else 0.0
        train_las_score = np.mean(train_las) if train_las else 0.0
        train_loss = np.mean(train_epoch_losses) if train_epoch_losses else 0.0
        train_uas_scores.append(train_uas_score)
        train_las_scores.append(train_las_score)
        train_losses.append(train_loss)

        model.eval()
        test_uas, test_las, test_epoch_losses = [], [], []
        logger.info(f"Epoch {epoch} - Testing")
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch} Test", total=len(test_loader)):
                wav = batch["wav"].to(device)
                words = batch["words"].to(device)
                pos = batch["pos"].to(device)
                heads = batch["heads"].to(device)
                rels = batch["rels"].to(device)
                start_times = batch["start_times"].to(device)
                end_times = batch["end_times"].to(device)

                wav_list = [wav[i].cpu().numpy() for i in range(wav.size(0))]
                inputs = processor(
                    wav_list,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                audio_feats = wav2vec2(**inputs).last_hidden_state
                arc_scores, rel_scores = model(words, pos, audio_feats, start_times, end_times)
                head_pred = eisner_decode(arc_scores)
                batch_size, seq_len = heads.size()
                rel_pred = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                for b in range(batch_size):
                    for i in range(seq_len):
                        if head_pred[b, i] < seq_len:
                            rel_pred[b, i] = rel_scores[b, i, head_pred[b, i]].argmax()

                loss = compute_loss(arc_scores, rel_scores, heads, rels, arc_loss_fn, rel_loss_fn, device)
                uas, las = compute_metrics(head_pred, rel_pred, heads, rels)
                test_uas.append(uas)
                test_las.append(las)
                test_epoch_losses.append(loss.item())

                write_conllu(
                    batch["ID"], batch["words"], batch["pos"], head_pred, rel_pred,
                    word_vocab, pos_vocab, rel_vocab, conllu_dir, "test"
                )

        test_uas_score = np.mean(test_uas) if test_uas else 0.0
        test_las_score = np.mean(test_las) if test_las else 0.0
        test_loss = np.mean(test_epoch_losses) if test_epoch_losses else 0.0
        test_uas_scores.append(test_uas_score)
        test_las_scores.append(test_las_score)
        test_losses.append(test_loss)
        logger.info(f"Epoch {epoch}, Train UAS: {train_uas_score:.4f}, LAS: {train_las_score:.4f}, Loss: {train_loss:.4f}, Test UAS: {test_uas_score:.4f}, LAS: {test_las_score:.4f}, Loss: {test_loss:.4f}")

        # Save metrics
        save_metrics(epoch, train_uas_score, train_las_score, test_uas_score, test_las_score, train_loss, test_loss, metrics_file)

        # Save current epoch model
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.ckpt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "test_uas": test_uas_score,
            "test_las": test_las_score
        }, checkpoint_path)
        logger.info(f"Saved current epoch model: {checkpoint_path}")

        # Delete previous epoch model
        if previous_checkpoint and os.path.exists(previous_checkpoint):
            try:
                os.remove(previous_checkpoint)
                logger.info(f"Deleted previous epoch model: {previous_checkpoint}")
            except Exception as e:
                logger.warning(f"Failed to delete previous epoch model: {str(e)}")
        previous_checkpoint = checkpoint_path

        # Save best model based on test_uas
        if test_uas_score > best_test_uas:
            best_test_uas = test_uas_score
            best_checkpoint = os.path.join(checkpoint_dir, "best_model.ckpt")
            shutil.copy(checkpoint_path, best_checkpoint)
            logger.info(f"Saved best model (Test UAS: {best_test_uas:.4f}): {best_checkpoint}")

        # Plot UAS, LAS, and Loss
        epoch_range = np.arange(1, epoch + 1)
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_range, train_uas_scores, label='Train UAS', marker='o')
        plt.plot(epoch_range, test_uas_scores, label='Test UAS', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('UAS')
        plt.title('UAS over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'uas_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_range, train_las_scores, label='Train LAS', marker='o')
        plt.plot(epoch_range, test_las_scores, label='Test LAS', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('LAS')
        plt.title('LAS over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'las_plot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(epoch_range, train_losses, label='Train Loss', marker='o')
        plt.plot(epoch_range, test_losses, label='Test Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plot_dir, 'loss_plot.png'))
        plt.close()

        logger.info(f"System RAM: {psutil.virtual_memory().percent}% used")
        if device.type == "cuda":
            logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            torch.cuda.empty_cache()

def main():
    check_system_resources()
    train_model()

def check_system_resources():
    for drive in ['D:\\']:
        disk = psutil.disk_usage(drive)
        if disk.free < 10 * 1e9:
            logger.warning(f"Low disk space: {disk.free / 1e9:.2f} GB free on {drive}")
    ram = psutil.virtual_memory()
    logger.info(f"System RAM: {ram.percent}% used, {ram.available / 1e9:.2f} GB free")
    if torch.cuda.is_available():
        logger.info(f"GPU Memory: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

if __name__ == "__main__":
    main()
