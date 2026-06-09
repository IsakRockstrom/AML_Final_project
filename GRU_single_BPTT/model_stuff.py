import matplotlib.pyplot as plt
import torch
from torch import nn

''' 
Train a model for cases:
1. No eval set
2. Eval set

Args:

eval (true/false),
train_loader,
eval_loader, 
X_eval_t,
y_eval_t

Params:

device,
nepochs,
n_features,
hidden_size,
num_layers,
drop_out,
lr (learning rate),
loss_fn,
'''

class Predictor(nn.Module):
    def __init__(self, n_features, hidden_size, num_layers, dropout, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, h_n = self.gru(x)
        y = self.head(out[:, -1, :]).squeeze(-1)  # [batch] if output_size=1, [batch, output_size] otherwise
        return y

def model_setup(device, params, n_epochs, train_loader, eval_loader, eval = False, X_eval_t = None, y_eval_lag = None):

    model = Predictor(
    params['n_features'],
    hidden_size=params['hidden_size'],
    num_layers=params['num_layers'],
    dropout=params['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    loss_fn = params['loss_fn']

    train_losses = []
    eval_losses = []

    for epoch in range(n_epochs):
        train_loss = 0
        model.train()
        for x_batch, y_batch in train_loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        eval_loss = 0

        if eval:
            
            model.eval()
            with torch.no_grad():
                for x_batch, y_batch in eval_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = model(x_batch)
                    loss = loss_fn(y_pred, y_batch)
                    eval_loss += loss.item()
            
            eval_losses.append(eval_loss / len(eval_loader))
            print(f"Epoch {epoch} with train loss: {train_loss / len(train_loader)} and eval loss: {eval_loss / len(eval_loader)}")

    if eval:

        fig, ax = plt.subplots()

        ax.plot(train_losses, label = "train")
        ax.plot(eval_losses, label = "validation")
        fig.legend()

        model.eval()
        with torch.no_grad():
            preds = model(X_eval_t.to(device)).cpu().numpy()

        plt.figure(figsize=(6, 6))
        plt.scatter(y_eval_lag, preds, s=2, alpha=0.3)

        lims = [min(y_eval_lag.min(), preds.min()),
                max(y_eval_lag.max(), preds.max())]
        plt.plot(lims, lims, 'r--', label='perfect prediction')

        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return model, train_losses, eval_losses