import torch


def train_linear(dataloader, model, loss_fn, optimizer, device, print_batch=1000000):
    """Performs one training epoch for a NNCV.

    Args:
        dataloader (Dataloader): Wrappper for the training set
        model (NNCV): The network to be trained
        loss_fn (function): Pytorch loss to be used during training
        optimizer (function): Pytorch optimizer to be used during training
        device (dvice): Pytorch device to run the calculation on. Supports CPU and GPU (cuda)
        print_batch (int, optional): Set to recieve printed updates on the lost every print_batch batches. Defaults to 1000000.

    Returns:
        float: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def train_egnn(loader, n_at, optimizer, loss_l1, model_1, device):
    """Function to perform one epoch of a GNN training.

    Args:
        loader (torch.utils.data.Dataloader): Wrapper around a GNNTrajectory dataset.
        n_at (int): Number of nodes per frame.
        optimizer (torch.optim): The optimizer object for the training.
        loss_l1 (torch.nn._Loss): Loss function for the training.
        model_1 (GNNCV): Graph-based model to be trained.
        device (str): Device that the training is performed on. (Required for GPU compatibility)

    Returns:
        float: Training loss at the end of the epoch.
    """
    res = {"loss": 0, "counter": 0}
    for batch, (X, y, r, c) in enumerate(loader):
        model_1.train()
        optimizer.zero_grad()
        batch_size = len(X)
        atom_positions = X.view(-1, 3).to(device)

        row_new = r.view(-1)
        col_new = c.view(-1)
        row_new = row_new[row_new > 0]
        col_new = col_new[col_new > 0]
        j = 0
        for i in range(1, len(row_new)):
            row_new[i] += j * n_at
            col_new[i] += j * n_at

            if row_new[i - 1] > row_new[i]:
                j += 1
                row_new[i] += n_at
                col_new[i] += n_at

        edges = [row_new.long().to(device), col_new.long().to(device)]
        label = y.to(device)

        pred = model_1(x=atom_positions, edges=edges, n_nodes=n_at)
        loss = loss_l1(pred, label)
        loss.backward()
        optimizer.step()

        res["loss"] += loss.item() * batch_size
        res["counter"] += batch_size

    return res["loss"] / res["counter"]


def train_perm(
    dataloader, model, loss_fn, optimizer, n_trans, device, print_batch=1000000
):
    """Performs one training epoch for a NNCV but the loss for each batch is not just calculated on one reference structure but a set of n_trans permutated versions of that structure.

    Args:
        dataloader (Dataloader): Wrapper around the training data
        model (NNCV): The model that is to be trained
        loss_fn (function): Pytorch loss function to be used for the training
        optimizer (function): Pytorch optimizer to be used during training
        n_trans (int): Number of permutated structures used for the loss calculations
        device (device): Pytorch device to run the calculations on. Supports CPU and GPU (cuda)
        print_batch (int, optional): Set to recieve printed updates on the loss every print_batches batches. Defaults to 1000000.

    Returns:
        float: Returns the last loss item. For easy learning curve recording. Alternatively one can use a Tensorboard.
    """
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.flatten(), y)

        for i in range(n_trans):
            # Shuffle the tensors in the batch
            pred = model(X[:, torch.randperm(X.size()[1])])
            loss += loss_fn(pred.flatten(), y)

        loss /= n_trans

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_batch == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    return loss.item()


def test(dataloader, model, loss_fn, device):
    """Calculates the current average test set loss.

    Args:
        dataloader (dataloader): Dataloader loading the test set
        model (NNCV): model that is being trained
        loss_fn (function): Pytorch loss function

    Returns:
        float: Current avg. test set loss
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred.flatten(), y).item()

    test_loss /= num_batches
    print(f"Avg. Test loss: {test_loss:>8f} \n")
    return test_loss
