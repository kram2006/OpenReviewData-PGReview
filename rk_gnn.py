import gc
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from transformers import AutoTokenizer, AutoModel

# ============================================================
# 0. MOUNT GOOGLE DRIVE
# ============================================================
# if not os.path.exists('/content/drive'):
#     from google.colab import drive
#     drive.mount('/content/drive')

# ============================================================
# 1. CONFIGURATION
# ============================================================
plm = {"scibert":"allenai/scibert_scivocab_uncased", "scidebertafull": "KISTI-AI/Scideberta-full", "scidebertacs":"KISTI-AI/scideberta-cs"}
for plmodel in plm.keys():
  #, "iclr_reviews_25", "neurips_reviews_25"
  for csv in ["iclr_reviews_24", "iclr_reviews_25", "iclr_reviews_26"]:
    print(csv, plmodel)
    # CSV_FILE = "/content/drive/MyDrive/"+csv+".csv"
    CSV_FILE = "./"+csv+".csv"
    MODEL_NAME = plm[plmodel]
    # EMBEDDING_FILE = "/content/drive/MyDrive/"+csv+"_"+plmodel+"_embeddings.dat"
    EMBEDDING_FILE = "./"+csv+"_"+plmodel+"_embeddings.dat"
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 128
    DROPOUT = 0.3
    RANDOM_SEED = 42

    # ============================================================
    # 2. RANDOM SEED
    # ============================================================

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    # ============================================================
    # 4. DEVICE
    # ============================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ============================================================
    # 5. READ CSV
    # ============================================================

    df = pd.read_csv(CSV_FILE, low_memory=False)
    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # ============================================================
    # 6. REQUIRED COLUMNS
    # ============================================================

    required_columns = [
        "submission_id",
        "title",
        "reviewer_id",
        "review_id",
        "summary",
        "soundness",
        "presentation",
        "contribution",
        "rating",
        "confidence"
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + str(missing))

    # ============================================================
    # 7. CLEAN DATA
    # ============================================================

    df = df[required_columns].copy()
    # Remove rows without required IDs
    df = df.dropna(subset=["submission_id", "reviewer_id", "review_id", "soundness", "presentation", "contribution", "confidence"])
    # Summary text
    df["summary"] = (df["summary"].fillna("").astype(str))
    # Numerical review attributes
    numeric_columns = ["rating"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    # Remove reviews without rating
    df = df.dropna(subset=["rating"])
    df = df.reset_index(drop=True)
    print("\nClean dataset:", df.shape)

    # ============================================================
    # 8. CREATE PAPER IDs
    # ============================================================

    paper_ids = (df["submission_id"].unique())
    paper_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    df["paper_idx"] = (df["submission_id"].map(paper_to_idx))

    # ============================================================
    # 9. CREATE REVIEWER IDs
    # ============================================================

    reviewer_ids = (df["reviewer_id"].unique())
    reviewer_to_idx = {reviewer_id: i for i, reviewer_id in enumerate(reviewer_ids)}
    df["reviewer_idx"] = (df["reviewer_id"].map(reviewer_to_idx))

    # ============================================================
    # 10. CREATE REVIEW IDs
    # ============================================================

    review_ids = (df["review_id"].unique())
    review_to_idx = {review_id: i for i, review_id in enumerate(review_ids)}
    df["review_idx"] = (df["review_id"].map(review_to_idx))
    print("\nNumber of papers:", len(paper_ids))
    print("Number of reviewers:", len(reviewer_ids))
    print("Number of reviews:", len(review_ids))

    # ============================================================
    # 11. LOAD SCI-BERT
    # ============================================================

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    print("\nModel loaded.")


    # ============================================================
    # 12. SCI-BERT EMBEDDING FUNCTION
    # ============================================================

    @torch.no_grad()
    def embed_batch(texts):
        encoded = tokenizer(texts, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        # --------------------------------------------------------
        # Mean pooling
        # --------------------------------------------------------
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        hidden = hidden * mask
        summed = hidden.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embeddings = (summed / counts)
        return embeddings.cpu()


    # ============================================================
    # 13. GENERATE / LOAD SCI-BERT EMBEDDINGS
    # ============================================================
    number_reviews = len(df)
    embedding_dimension = 768
    summary_embeddings = None # Initialize as None

    # Flag to determine if embeddings need to be generated
    should_generate = True

    if os.path.exists(EMBEDDING_FILE):
        try:
            # Attempt to open the file as a memory-mapped array in read mode
            # Use np.memmap directly for memory-mapped files
            temp_embeddings = np.memmap(EMBEDDING_FILE, dtype=np.float32, mode="r", shape=(number_reviews, embedding_dimension))

            # Check if the shape matches the expected shape
            if temp_embeddings.shape == (number_reviews, embedding_dimension):
                summary_embeddings = temp_embeddings
                print("\nUsing existing Model embeddings.")
                should_generate = False
            else:
                print("Existing embedding file has incorrect shape. Will regenerate embeddings.")
        except Exception as e:
            print(f"Error loading existing embeddings as memmap ({e}). Will regenerate embeddings.")
    else:
        print("\nEmbedding file not found. Will generate Model embeddings.")


    if should_generate:
        # If embeddings need to be generated or regenerated (due to error/wrong shape)
        print("\nGenerating Model embeddings...")
        # Open the file in write mode to create/overwrite
        summary_embeddings = np.memmap(
            EMBEDDING_FILE, dtype=np.float32, mode="w+", shape=(number_reviews, embedding_dimension))

        for start in tqdm(range(0, number_reviews, BATCH_SIZE)):
            end = min(start + BATCH_SIZE, number_reviews)
            texts_batch = df["summary"].iloc[start:end].tolist()
            try:
                emb = embed_batch(texts_batch)
                summary_embeddings[start:end] = emb.numpy()
                summary_embeddings.flush()
                # print(f"{end}/{number_reviews} embeddings generated")
                del emb
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as ex:
                print(f"Error at rows {start}-{end}:", str(ex))
                # Put zero vector for failed rows
                summary_embeddings[start:end] = 0 # This ensures the memmap is filled even on error
                summary_embeddings.flush()
        print("Embeddings saved to:", EMBEDDING_FILE)

    print("Embedding shape:", summary_embeddings.shape)

    # ============================================================
    # 14. CREATE HETEROGENEOUS GRAPH
    # ============================================================
    data = HeteroData()
    # ============================================================
    # PAPER NODES
    # ============================================================
    # Paper node features
    # We do NOT duplicate review embeddings here.
    #
    # Paper features can initially be zeros.  The GNN will learn paper representations from reviewer/review information.
    number_papers = len(paper_ids)
    data["paper"].x = torch.zeros((number_papers, 1), dtype=torch.float)

    # ============================================================
    # REVIEWER NODES
    # ============================================================
    number_reviewers = len(reviewer_ids)
    # Reviewer identity embedding
    #
    # Each reviewer starts with a learnable/unique  one-hot-like feature.
    data["reviewer"].x = torch.eye(number_reviewers, dtype=torch.float)
    # ============================================================
    # REVIEWER -> PAPER EDGES
    # ============================================================
    source = torch.tensor(df["reviewer_idx"].values, dtype=torch.long)
    target = torch.tensor(df["paper_idx"].values, dtype=torch.long)
    edge_index = torch.stack([source, target], dim=0)
    data["reviewer", "reviews", "paper"].edge_index = edge_index
    # ============================================================
    # PAPER -> REVIEWER REVERSE EDGES
    # ============================================================
    data["paper", "rev_by", "reviewer"].edge_index = edge_index.flip(0)
    # ============================================================
    # 15. REVIEW FEATURES
    # ============================================================
    # For every review:
    #
    # model(summary) = 768

    review_numeric = df[["rating"]].fillna(0).values
    review_numeric = torch.tensor(review_numeric, dtype=torch.float)
    # Ensure summary_embeddings is not None and is a numpy array
    if summary_embeddings is None:
        raise ValueError("Summary embeddings could not be loaded or generated.")
    # Create a writable copy to avoid UserWarning from PyTorch
    summary_tensor = torch.from_numpy(np.asarray(summary_embeddings).copy()).float()
    review_edge_features = torch.cat([summary_tensor, review_numeric], dim=1)
    print("\nReview edge feature shape:", review_edge_features.shape)

    # ============================================================
    # 16. STORE REVIEW FEATURES
    # ============================================================
    data["reviewer", "reviews", "paper"].edge_attr = (review_edge_features)
    data["paper", "rev_by", "reviewer"].edge_attr = (review_edge_features)
    # ============================================================
    # 17. PAPER TARGET = RATING
    # ============================================================

    # Multiple reviewers can rate the same paper. We calculate the mean rating for each paper.
    paper_rating = (df.groupby("paper_idx")["rating"].mean())
    rating_target = torch.zeros(number_papers, dtype=torch.float)
    for paper_idx, rating in (paper_rating.items()):
        rating_target[int(paper_idx)] = float(rating)
    data["paper"].y = rating_target

    # ============================================================
    # 18. PRINT GRAPH
    # ============================================================

    print("\n==============================")
    print("HETEROGENEOUS GRAPH")
    print("==============================")
    print(data)

    print("\nPaper nodes:", data["paper"].x.shape)
    print("Reviewer nodes:", data["reviewer"].x.shape)
    print("Review edges:", data["reviewer", "reviews", "paper"].edge_index.shape)
    print("Review features:", data["reviewer", "reviews", "paper"].edge_attr.shape)

    # ============================================================
    # 19. MOVE GRAPH TO GPU
    # ============================================================

    data = data.to(device)

    # ============================================================
    # 20. HETEROGENEOUS GNN
    # ============================================================
    class ReviewGNN(nn.Module):
        def __init__(self, reviewer_input, paper_input, edge_dim, hidden_dim=128):
            super().__init__()
            # ----------------------------------------------------
            # Reviewer feature projection
            # ----------------------------------------------------
            self.reviewer_encoder = nn.Linear(reviewer_input, hidden_dim)
            # ----------------------------------------------------
            # Paper feature projection
            # ----------------------------------------------------
            self.paper_encoder = nn.Linear(paper_input, hidden_dim)
            # ----------------------------------------------------
            # Edge feature projection
            # ----------------------------------------------------
            self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
            # ----------------------------------------------------
            # GNN layers
            # ----------------------------------------------------
            self.conv1 = HeteroConv({
                ("reviewer", "reviews", "paper"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("paper", "rev_by", "reviewer"): SAGEConv((hidden_dim, hidden_dim), hidden_dim)}, aggr="mean")
            self.conv2 = HeteroConv({
                ("reviewer", "reviews", "paper"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
                ("paper", "rev_by", "reviewer"): SAGEConv((hidden_dim, hidden_dim), hidden_dim)}, aggr="mean")
            # ----------------------------------------------------
            # Rating prediction
            # ----------------------------------------------------
            self.rating_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(DROPOUT),
                                            nn.Linear(64, 1))

        def forward(self, x_dict, edge_index_dict):
            # Encode initial node features
            x_dict = {"paper": self.paper_encoder(x_dict["paper"]),
                      "reviewer": self.reviewer_encoder(x_dict["reviewer"])}
            # ----------------------------------------------------
            # GNN Layer 1
            # ----------------------------------------------------
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}
            # ----------------------------------------------------
            # GNN Layer 2
            # ----------------------------------------------------
            x_dict = self.conv2(x_dict, edge_index_dict)
            x_dict = {key: F.relu(value) for key, value in x_dict.items()}
            # ----------------------------------------------------
            # Paper representation
            # ----------------------------------------------------
            paper_embedding = (x_dict["paper"])
            # ----------------------------------------------------
            # Rating
            # ----------------------------------------------------
            rating = self.rating_head(paper_embedding)
            return (paper_embedding, rating.squeeze(-1))


    # ============================================================
    # 21. CREATE MODEL
    # ============================================================
    model = ReviewGNN(reviewer_input=data["reviewer"].x.shape[1], paper_input=data["paper"].x.shape[1],
                      edge_dim=data["reviewer", "reviews", "paper"].edge_attr.shape[1],
                      hidden_dim=HIDDEN_DIM).to(device)
    # ============================================================
    # 22. OPTIMIZER
    # ============================================================
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()
    # ============================================================
    # 23. TRAIN / TEST SPLIT
    # ============================================================
    number_papers = (data["paper"].num_nodes)
    indices = torch.randperm(number_papers, device=device)
    train_size = int(0.8 * number_papers)
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    print("\nTraining papers:", len(train_idx))
    print("Testing papers:", len(test_idx))

    # ============================================================
    # 24. TRAINING
    # ============================================================

    print("\nStarting training...")

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        optimizer.zero_grad()
        paper_embedding, prediction = model(data.x_dict, data.edge_index_dict)
        loss = loss_function(prediction[train_idx], data["paper"].y[train_idx])
        loss.backward()
        optimizer.step()
        # --------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------
        model.eval()
        with torch.no_grad():
            _, prediction = model(data.x_dict, data.edge_index_dict)
    train_loss = loss_function(prediction[train_idx], data["paper"].y[train_idx])
    test_loss = loss_function(prediction[test_idx], data["paper"].y[test_idx])
    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | "
              f"Train Loss: {train_loss.item():.4f} | "
              f"Test Loss: {test_loss.item():.4f}"
              )

    # ============================================================
    # 25. FINAL PREDICTIONS AND EVALUATION
    # ============================================================
    import time
    start = time.time()
    model.eval()
    print("Time : ",time.time() - start)
    with torch.no_grad():
        paper_embeddings_gnn, predictions = model(data.x_dict, data.edge_index_dict)

    # Move to CPU / NumPy
    predictions = predictions.cpu().numpy()
    paper_embeddings_gnn = paper_embeddings_gnn.cpu().numpy()
    actual_ratings = data["paper"].y.cpu().numpy()
    # ------------------------------------------------------------
    # TEST SET PREDICTIONS
    # ------------------------------------------------------------
    y_test = actual_ratings[test_idx.cpu().numpy()]
    pred_test = predictions[test_idx.cpu().numpy()]
    # ============================================================
    # 26. REGRESSION METRICS
    # ============================================================
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, pred_test)
    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)
    rmse = np.sqrt(mse)
    print("\n========================================")
    print("TEST SET PERFORMANCE", csv, plmodel)
    print("========================================")
    print(f"MSE  : {mse:.4f}\tMAE  : {mae:.4f}\tR²   : {r2:.4f}\tRMSE : {rmse:.4f}")
    # ============================================================
    # 27. SAVE GNN PAPER EMBEDDINGS
    # ============================================================
    # np.save("/content/drive/MyDrive/"+csv+"_"+plmodel+"_gnn_paper_embeddings.npy", paper_embeddings_gnn)
    np.save("./" + csv + "_" + plmodel + "_gnn_paper_embeddings.npy", paper_embeddings_gnn)
    # ============================================================
    # 28. SAVE PREDICTIONS
    # ============================================================
    result = pd.DataFrame({"submission_id": paper_ids, "actual_rating": actual_ratings, "predicted_rating": predictions})
    # Add rounded prediction
    result["predicted_rating_rounded"] = np.rint(result["predicted_rating"]).astype(int)
    # result.to_csv("/content/drive/MyDrive/"+csv+"_"+plmodel+"__gnn_predictions.csv", index=False)
    result.to_csv("./" + csv + "_" + plmodel + "__gnn_predictions.csv", index=False)
    print("\nResults saved.")
    print(result.head(10))
